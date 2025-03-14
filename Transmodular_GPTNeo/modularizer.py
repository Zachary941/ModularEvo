"""This project aims to modularize GPT-Neo."""

import argparse
import os.path
import logging
import sys
from datetime import datetime
import torch
import math
import copy
from torch.optim import AdamW
from datasets import load_dataset, load_from_disk
from transformers import TrainerCallback,DataCollatorForLanguageModeling, TrainingArguments, Trainer, get_linear_schedule_with_warmup,EarlyStoppingCallback
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from torch.utils.tensorboard import SummaryWriter
from mask_layer import MaskLinear, Binarization, init_mask_model

class StopOnWRRCallback(TrainerCallback):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = 0.25

    def on_log(self, args, state, control, logs=None, **kwargs):
        current_wrr = logs.get('wrr')
        if current_wrr is not None and current_wrr <= self.threshold:
            logging.info(f"WRR arrive  :{current_wrr:.2%},stop training")
            control.should_save = True
            control.should_training_stop = True

def preprocess_pile(raw_dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding=True)
    
    pile_data = raw_dataset.map(tokenize_function, batched=True, remove_columns=raw_dataset['train'].column_names,
                                desc='preprocess', num_proc=8)

    pile_data['train'] = pile_data['train'].add_column('labels', pile_data['train']['input_ids'])
    pile_data['test'] = pile_data['test'].add_column('labels', pile_data['test']['input_ids'])
    return pile_data

def clip_processed_data(processed_data, n_train, n_test):
    part_processed_data = copy.deepcopy(processed_data)
    if n_train > 0:
        part_train_data = processed_data['train'].select(list(range(n_train)))
        part_processed_data['train'] = part_train_data

    if n_test > 0:
        part_test_data = processed_data['test'].select(list(range(n_test)))
        part_processed_data['test'] = part_test_data
    logging.info(part_processed_data)
    return part_processed_data


class Modularizer(Trainer):

    def __init__(self, alpha, non_bin, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.count_training_step = 0
        self.final_loss_tr = 0.0
        self.loss_clm_tr = 0.0
        self.wrr = 1.0
        self.non_bin = non_bin

    def compute_wrr(self, detailed=False):
        masks = []
        for n, layer in self.model.named_modules():
            if hasattr(layer, 'weight_mask'):
                masks.append(torch.flatten(layer.weight_mask))
                if layer.bias_mask is not None:
                    masks.append(torch.flatten(layer.bias_mask))

        if detailed:
            output_layer_mask = masks[-1]
            assert len(output_layer_mask) == 768 * 50257  # because the dimension of lm_head is this
            bin_output_layer_mask = Binarization.apply(output_layer_mask)
            output_layer_wrr = torch.mean(bin_output_layer_mask)

            feature_layer_mask = torch.cat(masks[:-1], dim=0)
            bin_feature_layer_mask = Binarization.apply(feature_layer_mask)
            feature_layer_wrr = torch.mean(bin_feature_layer_mask)

        masks = torch.cat(masks, dim=0)
        bin_masks = Binarization.apply(masks)
        wrr = torch.mean(bin_masks)
        if self.non_bin:
            loss_wrr = torch.mean(torch.nn.functional.hardtanh(masks, min_val=0, max_val=1))
        else:
            loss_wrr = wrr

        if detailed:
            return loss_wrr, wrr, feature_layer_wrr, output_layer_wrr
        else:
            return loss_wrr, wrr

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Copy from Trainer.compute_loss()
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # Custom: add weight retention rate
        detailed = True
        if detailed:
            loss_wrr, wrr, feature_layer_wrr, output_layer_wrr = self.compute_wrr(detailed=detailed)
        else:
            loss_wrr, wrr = self.compute_wrr()
        self.wrr = wrr
        self.log({'wrr': self.wrr.item()})
        final_loss = loss + self.alpha * loss_wrr

        if model.training:
            self.final_loss_tr += final_loss.item()
            self.loss_clm_tr += loss.item()
            self.count_training_step += 1
            writer.add_scalar(f'Modular/loss_Total', self.final_loss_tr / self.count_training_step, self.count_training_step)
            writer.add_scalar(f'Modular/loss_CLM', self.loss_clm_tr / self.count_training_step, self.count_training_step)
            writer.add_scalar(f'Modular/loss_WRR', loss_wrr, self.count_training_step)
            writer.add_scalar(f'Modular/WRR', wrr, self.count_training_step)
            if detailed:
                writer.add_scalar(f'Modular/WRR_output_layer', output_layer_wrr, self.count_training_step)
                writer.add_scalar(f'Modular/WRR_feature_layer', feature_layer_wrr, self.count_training_step)
        else:
            # NOTE: in evaluation, only output the loss of CLM. the wrr and loss of wrr are the same as the last step of training.
            final_loss = loss

        return (final_loss, outputs) if return_outputs else final_loss


def load_data(dataset_path_train, dataset_path_test):

    data_file_gpt = {"train": dataset_path_train, "test": dataset_path_test}

    pile_data = load_dataset('json', data_files=data_file_gpt)

    logging.info("+++++++++++++++ Raw +++++++++++++++")
    logging.info(f"pile_data type:{type(pile_data)}")
    logging.info(pile_data["test"][0])
    logging.info("===================================")
    logging.info(pile_data)
    return pile_data


def get_data_path(lang, is_raw):
    if is_raw:
        if lang == 'python':
            dataset_path_train = f'data/dataset/pile/train/Github_train/python_train.jsonl'
            dataset_path_test = f'data/dataset/pile/test/Github_test/python_test.jsonl'
        elif lang == 'law':
            dataset_path_train = f'data/dataset/pile/train/FreeLaw_train.jsonl'
            dataset_path_test = f'data/dataset/pile/test/FreeLaw_test.jsonl'
        elif lang == 'github':
            dataset_path_train = f'data/dataset/pile/train/Github_train/Github_train.jsonl'
            dataset_path_test = f'data/dataset/pile/test/Github_test/Github_test.jsonl'
        elif lang == 'math':
            dataset_path_train = f'data/dataset/pile/train/math_train.jsonl'
            dataset_path_test = f'data/dataset/pile/test/math_test.jsonl'
        else:
            raise ValueError
        return dataset_path_train, dataset_path_test

    else:
        if lang == 'python':
            dataset_path = f'data/dataset/pile/train/Github_train/python_processed'
        elif lang == 'law':
            dataset_path = f'data/dataset/pile/train/FreeLaw_processed'
        elif lang == 'github':
            dataset_path = f'data/dataset/pile/train/Github_train/github_processed'
        else:
            raise ValueError
        return dataset_path


def get_preprocessed_data(lang, tokenizer_gpt):
    path_lang_data = get_data_path(lang=lang, is_raw=False)
    if os.path.exists(path_lang_data):
        logging.info(f'Loading preprocessed data from {path_lang_data}')
        processed_lang_data = load_from_disk(path_lang_data)
    else:
        dataset_path_train, dataset_path_test = get_data_path(lang=lang, is_raw=True)
        pile_data = load_data(dataset_path_train, dataset_path_test)
        processed_lang_data = preprocess_pile(pile_data, tokenizer_gpt)
        logging.info(f'Saving preprocessed data to {path_lang_data}')
        processed_lang_data.save_to_disk(path_lang_data)
    return processed_lang_data


def main():
    model_path_gpt = f"./data/gpt-neo-125m"
    model_gpt = GPTNeoForCausalLM.from_pretrained(model_path_gpt)
    module = init_mask_model(model=model_gpt, no_mask=[])
    logging.info(f'\n\n{module}\n\n')

    tokenizer_gpt = GPT2Tokenizer.from_pretrained(model_path_gpt)

    tokenizer_gpt.pad_token = tokenizer_gpt.eos_token
    processed_lang_data = get_preprocessed_data(arguments.lang, tokenizer_gpt)

    logging.info("+++++++++++++++ Processed +++++++++++++++")
    logging.info(processed_lang_data["test"]['input_ids'][0])
    logging.info(processed_lang_data["test"]['labels'][0])
    logging.info("===================================")
    logging.info(processed_lang_data)

    clipped_lang_data = clip_processed_data(processed_lang_data, n_train=arguments.num_train_samples, n_test=1000)
    logging.info(f'Clipped data:\n{clipped_lang_data}\n')


    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer_gpt,
        mlm=False, 
    )

    mask_names = ['weight_mask', 'bias_mask']
    mask_params = [p for n, p in module.named_parameters() if any(mn in n for mn in mask_names)]
    optimizer = AdamW(mask_params, lr=arguments.lr, weight_decay=arguments.weight_decay)

    module_save_dir = f'data/module_{arguments.lang}/lr_{arguments.lr}_alpha_{arguments.alpha}_bs_{arguments.batch_size}'
    if arguments.non_bin:
        module_save_dir = f'{module_save_dir}_nonbin'

    training_args = TrainingArguments(
        output_dir=module_save_dir,
        do_train=True,
        evaluation_strategy='steps',
        eval_steps=200,
        logging_steps=10,
        per_device_train_batch_size=arguments.batch_size,
        per_device_eval_batch_size=arguments.batch_size,
        learning_rate=arguments.lr,
        num_train_epochs=arguments.n_epochs,
        weight_decay=arguments.weight_decay,
        save_steps=3000,
        save_total_limit=1,
        seed=1203,
        dataloader_num_workers=2,
        # warmup_ratio=0.1,
        warmup_steps=2000,
        fp16=arguments.fp16,
        load_best_model_at_end=True # load the best model when training ends
    )

    modularizer = Modularizer(
        model=module,
        args=training_args,
        train_dataset=clipped_lang_data['train'],
        eval_dataset=clipped_lang_data['test'],
        data_collator=data_collator,
        optimizers=(optimizer, None),
        alpha=arguments.alpha,
        non_bin=arguments.non_bin,
        callbacks=[StopOnWRRCallback(threshold=0.25)]
    )

    if arguments.do_train:
        # evaluate original model on test dataset.
        trainer = Trainer(
            model=model_gpt,
            args=training_args,
            train_dataset=clipped_lang_data['train'],
            eval_dataset=clipped_lang_data['test'],
            data_collator=data_collator,
        )
        model_eval_results = trainer.evaluate()
        logging.info(f"[PART TEST]")
        logging.info(f"Model CE Loss: {model_eval_results['eval_loss']}")
        logging.info(f"Model Perplexity: {math.exp(model_eval_results['eval_loss']):.2f}\n\n")

        modularizer.train()
        modularizer.save_model(f'{module_save_dir}/result')

        logging.info('=' * 100)
        logging.info(f'WRR: {modularizer.wrr:.2%}')

        trainer.eval_dataset = processed_lang_data['test']
        model_eval_full_test = trainer.evaluate()

        modularizer.eval_dataset = processed_lang_data['test']
        module_eval_full_test = modularizer.evaluate()

        # Since the masked tokens could be different in each step, the eval_results could be different in every evaluation.
        logging.info(f"[FULL TEST]")
        logging.info(f"+ Module CE Loss: {module_eval_full_test['eval_loss']:.4f}")
        logging.info(f"+ Module Perplexity: {math.exp(module_eval_full_test['eval_loss']):.2f}")
        logging.info(f"- Model CE Loss: {model_eval_full_test['eval_loss']:.4f}")
        logging.info(f"- Model Perplexity: {math.exp(model_eval_full_test['eval_loss']):.2f}")

        if arguments.other_lang is not None:
            processed_other_lang_data = get_preprocessed_data(arguments.other_lang, tokenizer_gpt)

            logging.info("+++++++++++++++ Processed +++++++++++++++")
            logging.info(processed_other_lang_data["test"]['input_ids'][0])
            logging.info(processed_other_lang_data["test"]['labels'][0])
            logging.info("===================================")
            logging.info(processed_other_lang_data)

            trainer.train_dataset = processed_other_lang_data['train']
            trainer.eval_dataset = processed_other_lang_data['test']
            model_eval_results_on_other_lang = trainer.evaluate()

            modularizer.train_dataset = processed_other_lang_data['train']
            modularizer.eval_dataset = processed_other_lang_data['test']
            module_eval_results_on_other_lang = modularizer.evaluate()

            logging.info(f'\n================Summary================')
            logging.info(f'WRR: {modularizer.wrr:.2%}\n')
            logging.info(f"[On {arguments.lang}]")
            logging.info(f"+ Module CE Loss: {module_eval_full_test['eval_loss']:.4f}")
            logging.info(f"+ Module Perplexity: {math.exp(module_eval_full_test['eval_loss']):.2f}")
            logging.info(f"- Model CE Loss: {model_eval_full_test['eval_loss']:.4f}")
            logging.info(f"- Model Perplexity: {math.exp(model_eval_full_test['eval_loss']):.2f}")

            logging.info(f'\n[On {arguments.other_lang}]')
            logging.info(f"+ Module CE Loss: {module_eval_results_on_other_lang['eval_loss']:.4f}")
            logging.info(f"+ Module Perplexity: {math.exp(module_eval_results_on_other_lang['eval_loss']):.2f}")
            logging.info(f"- Model CE Loss: {model_eval_results_on_other_lang['eval_loss']:.4f}")
            logging.info(f"- Model Perplexity: {math.exp(model_eval_results_on_other_lang['eval_loss']):.2f}")

    elif arguments.do_eval:  # TODO: fix. TO load the checkpoint, i.e., the resulting module.
        module_eval_results = modularizer.evaluate()
        logging.info(f'Module: {module_eval_results}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', choices=["python", "law", "github"], required=True)
    parser.add_argument('--other_lang', choices=["python", "law", "github"],
                        help="evaluate the resulting module on other domain.")
    parser.add_argument('--do_train', action='store_true')
    # parser.add_argument('--do_eval', action='store_true')  # evaluate on test dataset. TO do
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--n_epochs', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--non_bin', action='store_true',
                        help='When computing the loss of weight retention, using the continuous mask with hardtanh(0, 1) '
                             'rather than using the binarized mask with Binarization()')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_train_samples', type=int, default=-1)
    parser.add_argument('--fp16', action='store_true')
    arguments = parser.parse_args()

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"./log/modularizer_{arguments.lang}_lr_{arguments.lr}_alpha_{arguments.alpha}_bs_{arguments.batch_size}_ne_{arguments.n_epochs}_{current_time}.log"
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_filename),
                        logging.StreamHandler(sys.stdout)
                    ])
    logging.info(arguments)

    if arguments.other_lang is not None:
        assert arguments.lang != arguments.other_lang

    tensorboard_dir = f'./tensorboard_log/modular/' \
                      f'{arguments.lang}/' \
                      f'lr_{arguments.lr}_alpha_{arguments.alpha}_bs_{arguments.batch_size}_ne_{arguments.n_epochs}_' \
                      f'nsample_{arguments.num_train_samples}'
    if arguments.non_bin:
        tensorboard_dir = f'{tensorboard_dir}_nonbin'
    if arguments.fp16:
        tensorboard_dir = f'{tensorboard_dir}_fp16'

    logging.info(f'tensorboard dir: {tensorboard_dir}')
    writer = SummaryWriter(tensorboard_dir)
    main()
