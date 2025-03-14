import argparse
import torch
import os
from torch.optim import AdamW
from transformers import RobertaTokenizer, RobertaForMaskedLM, pipeline
from datasets import load_dataset, load_from_disk
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer, EarlyStoppingCallback
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from torch.utils.tensorboard import SummaryWriter
from mask_layer import MaskLinear, Binarization, init_mask_model


def preprocess(raw_dataset, tokenizer):
    def _filter(examples):
        filtered_data = []
        for func_name, code_tokens, doc_tokens in zip(examples['func_name'], examples['func_code_tokens'],
                                                      examples['func_documentation_tokens']):
            if 'test' in func_name:  # CodeBERT: "function names with substring “test” are removed."
                continue

            if len(code_tokens) < 20:  # CodeBERT: "functions shorter than three lines are removed,"
                continue

            if len(doc_tokens) < 3:  # CodeBERT: " documentations shorter than three tokens are removed,":
                continue

            filtered_data.append(' '.join(code_tokens + doc_tokens))
        results = {'filtered_data': filtered_data}
        return results

    def _tokenize_and_label(examples):
        tokenized_data = tokenizer(examples['filtered_data'], truncation=True, padding=True)
        # tokenized_data['labels'] = tokenized_data['input_ids'].copy()
        return tokenized_data

    filtered_data = raw_dataset.map(_filter, batched=True, num_proc=8,
                                    remove_columns=raw_dataset["train"].column_names, desc='filtering')
    preprocessed_data = filtered_data.map(_tokenize_and_label, batched=True, num_proc=8,
                                          remove_columns=filtered_data['train'].column_names, desc='tokenizing')
    return preprocessed_data


class Modularizer(Trainer):
    def __init__(self, alpha, low_rank, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.count_training_step = 0
        self.final_loss_tr = 0.0
        self.loss_mlm_tr = 0.0
        self.wrr = 1.0
        self.low_rank = low_rank

    def compute_wrr(self):
        masks = []
        if self.low_rank:
            for n, layer in self.model.named_modules():
                if hasattr(layer, 'weight_mask_A'):
                    weight_mask = layer.weight_mask_A @ layer.weight_mask_B
                    masks.append(torch.flatten(weight_mask))
                    if layer.bias_mask is not None:
                        masks.append(torch.flatten(layer.bias_mask))
        else:
            for n, layer in self.model.named_modules():
                if hasattr(layer, 'weight_mask'):
                    masks.append(torch.flatten(layer.weight_mask))
                    if layer.bias_mask is not None:
                        masks.append(torch.flatten(layer.bias_mask))
        masks = torch.cat(masks, dim=0)
        bin_masks = Binarization.apply(masks)
        wrr = torch.mean(bin_masks)
        loss_wrr = wrr
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
        # wrr = self.compute_wrr()
        loss_wrr, wrr = self.compute_wrr()
        self.wrr = wrr
        # print(f'loss_wrr: {loss_wrr:.2%},loss:{loss}')
        loss = loss.mean()
        final_loss = loss + self.alpha * loss_wrr

        if model.training:
            self.final_loss_tr += final_loss.item()
            self.loss_mlm_tr += loss.item()
            self.count_training_step += 1

            writer.add_scalar(f'Modular/loss_Total', self.final_loss_tr / self.count_training_step, self.count_training_step)
            writer.add_scalar(f'Modular/loss_MLM', self.loss_mlm_tr / self.count_training_step, self.count_training_step)
            writer.add_scalar(f'Modular/loss_WRR', loss_wrr, self.count_training_step)
            writer.add_scalar(f'Modular/WRR', wrr, self.count_training_step)
        else:
            # NOTE: in evaluation, only output the loss of MLM. the wrr and loss of wrr are the same as the last step of training.
            final_loss = loss

        return (final_loss, outputs) if return_outputs else final_loss


class MyEarlyStoppingCallback(EarlyStoppingCallback):
    def __init__(self, model_val_mlm_loss, early_stopping_patience=3, early_stopping_threshold=0.001):
        super().__init__(early_stopping_patience, early_stopping_threshold)
        self.model_val_mlm_loss = model_val_mlm_loss

    def on_train_begin(self, args, state, control, **kwargs):
        pass

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        module_val_mlm_loss = metrics.get('eval_loss')

        if module_val_mlm_loss is None:
            raise ValueError

        writer.add_scalar(f'Modular/val_mlm_loss', module_val_mlm_loss, state.global_step)
        if module_val_mlm_loss > self.model_val_mlm_loss:
            print(f'\nWarning: module {module_val_mlm_loss} > model {self.model_val_mlm_loss} \n')
            # control.should_training_stop = True  # TEST

def main():
    model_path = f'./data/pretrain_model/codebert-base-mlm'
    if not os.path.exists(model_path):
        model_path = 'microsoft/codebert-base-mlm'

    model = RobertaForMaskedLM.from_pretrained(model_path)
    module = init_mask_model(model=model, no_mask=['lm_head'], is_low_rank=arguments.low_rank, rank=arguments.rank)
    print(f'\n\n{module}\n\n')
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    tokenizer.do_lower_case = True

    dataset_path = f'data/dataset/code_search_net/dataset/{arguments.lang}'
    processed_dataset_path = f'{dataset_path}_processed'
    if os.path.exists(processed_dataset_path):
        preprocessed_dataset = load_from_disk(processed_dataset_path)
    else:
        if not os.path.exists(dataset_path):
            # the second argument can be "all", "go", "java", "javascript", "php", "python", "ruby"
            code_search_net = load_dataset("code_search_net", arguments.lang)
            code_search_net.save_to_disk(dataset_path)
        code_search_net = load_from_disk(dataset_path)

        preprocessed_dataset = preprocess(code_search_net, tokenizer)
        preprocessed_dataset.save_to_disk(processed_dataset_path)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    mask_names = ['weight_mask', 'bias_mask']
    mask_param_names = [n for n, p in module.named_parameters() if any(mn in n for mn in mask_names)]
    print('NOTE: this is a new version. lm_head is trained during modularizing.')
    lm_head_param_names = [n for n, p in module.lm_head.named_parameters()]

    print(f'Trainable Parameters:')
    print('='*50)
    print(mask_param_names)
    print('-'*50)
    print(lm_head_param_names, "\n")

    mask_params = [p for n, p in module.named_parameters() if any(mn in n for mn in mask_names)]
    # NOTE: cannot directly find the keyword "lm_head" in module.named_parameters(), as the decoder in lm_head is not returned by module.named_parameters(). 
    # here need to use module.lm_head.named_parameters()
    lm_head_params = [p for n, p in module.lm_head.named_parameters()]

    # optimizer = AdamW(mask_params, lr=arguments.lr, weight_decay=arguments.weight_decay)
    optimizer = AdamW(
        [{'params': mask_params, 'lr': arguments.lr},
         {'params': lm_head_params, 'lr': arguments.lr_lmhead}], 
        weight_decay=arguments.weight_decay)

    module_save_dir = f'data/module_{arguments.lang}/lr_{arguments.lr}_alpha_{arguments.alpha}_ne_{arguments.n_epochs}'
    if arguments.low_rank:
        module_save_dir = f'{module_save_dir}_rank_{arguments.rank}_lrlm_{arguments.lr_lmhead}'
    if arguments.early_stop:
        module_save_dir = f'{module_save_dir}_early_{arguments.eval_steps}'
    if arguments.warmup_ratio != 0.1:
        module_save_dir = f'{module_save_dir}_warm_{arguments.warmup_ratio}'

    training_args = TrainingArguments(
        output_dir=module_save_dir,
        do_train=True,
        evaluation_strategy='steps' if arguments.early_stop else 'epoch',
        logging_steps=arguments.eval_steps,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=arguments.lr,
        num_train_epochs=arguments.n_epochs,
        weight_decay=arguments.weight_decay,
        save_steps=arguments.eval_steps if arguments.early_stop else 3000,
        save_total_limit=2,
        seed=1203,
        dataloader_num_workers=4,
        warmup_ratio=arguments.warmup_ratio,
    )
    
    if arguments.do_train:
        # evaluate original model on test dataset.
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=preprocessed_dataset['train'],
            eval_dataset=preprocessed_dataset['test'],
            data_collator=data_collator,
        )
        model_eval_results = trainer.evaluate()
        print(f'Model MLM Loss: {model_eval_results}\n\n')

        # modularize
        modularizer = Modularizer(
            model=module,
            args=training_args,
            train_dataset=preprocessed_dataset['train'],
            eval_dataset=preprocessed_dataset['test'],
            data_collator=data_collator,
            optimizers=(optimizer, None),
            alpha=arguments.alpha,
            low_rank=arguments.low_rank,
            callbacks=[MyEarlyStoppingCallback(model_val_mlm_loss=model_eval_results['eval_loss'])] if arguments.early_stop else None
        )

        modularizer.train()
        modularizer.save_model(f'{module_save_dir}/result')

        print('=' * 100)
        print(f'WRR: {modularizer.wrr:.2%}')
        module_eval_results = modularizer.evaluate()
        # Since the masked tokens could be different in each step, the eval_results could be different in every evaluation.
        print(f'Module MLM Loss: {module_eval_results}\n')
        print(f'Model  MLM Loss: {model_eval_results}\n\n')
        torch.save(modularizer.model.state_dict(), f'{module_save_dir}/result/pytorch_model_try.bin')
        torch.save(module.state_dict(), f'{module_save_dir}/result/pytorch_model.bin')
    # elif arguments.do_eval:  # TODO: fix. TO load the checkpoint, i.e., the resulting module.
    #     module_eval_results = modularizer.evaluate()
    #     print(f'Module: {module_eval_results}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', choices=["go", "java", "javascript", "php", "python", "ruby"], required=True)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')  # evaluate on test dataset.
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--n_epochs', type=int, default=2)  # ruby: 13, go: 2, java: 2, javascript: 7, python: 2, php: 2
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--low_rank', action='store_true')
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--lr_lmhead', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--early_stop', action='store_true',
                        help='whether to early stop modularization '
                             'when the validation mlm loss of module is lower than that of model')
    parser.add_argument('--eval_steps', type=int, default=100)
    arguments = parser.parse_args()
    print(arguments)

    tensorboard_dir = f'./tensorboard_log/modular/{arguments.lang}/lr_{arguments.lr}_alpha_{arguments.alpha}_ne_{arguments.n_epochs}'
    if arguments.low_rank:
        assert arguments.rank > 0
        tensorboard_dir = f'{tensorboard_dir}_rank_{arguments.rank}_lrlm_{arguments.lr_lmhead}'
    if arguments.early_stop:
        tensorboard_dir = f'{tensorboard_dir}_early_{arguments.eval_steps}'
    if arguments.warmup_ratio != 0.1:
        tensorboard_dir = f'{tensorboard_dir}_warm_{arguments.warmup_ratio}'

    print(f'tensorboard dir: {tensorboard_dir}')
    writer = SummaryWriter(tensorboard_dir)
    main()
