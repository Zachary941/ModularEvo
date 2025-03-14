import argparse
import torch
import os
from torch.optim import AdamW
from transformers import RobertaTokenizer, RobertaForMaskedLM, pipeline, T5Tokenizer,T5ForConditionalGeneration,DataCollatorForSeq2Seq,Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset, load_from_disk
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer, EarlyStoppingCallback
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from torch.utils.tensorboard import SummaryWriter
from mask_layer import MaskLinear, Binarization, init_mask_model
from transformers import TrainerCallback, TrainerState, TrainerControl
import logging
import sys
from datetime import datetime
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def check_dataset(dataset):
    for split in ['train', 'validation', 'test']:
        if split not in dataset:
            continue
        for index, sample in enumerate(dataset[split]):
            for key in ['input_ids', 'labels']:
                if key not in sample or sample[key] is None:
                    logging.info(f"Found None in dataset[{split}][{index}]['{key}']")
                    return False
                if len(sample[key]) == 0:
                    logging.info(f"Found empty list in dataset[{split}][{index}]['{key}']")
                    return False
    logging.info("Dataset check passed.")
    return True
# class StopOnWRRCallback(TrainerCallback):
#     def __init__(self, threshold):
#         super().__init__()
#         self.threshold = 0.25

#     def on_log(self, args, state, control, logs=None, **kwargs):
#         current_wrr = logs.get('wrr')
#         if current_wrr is not None and current_wrr <= self.threshold:
#             logging.info(f"WRR 达到阈值 {current_wrr:.2%}，停止训练并保存模型。")
#             control.should_save = True
#             control.should_training_stop = True

class StopOnWRRCallback(TrainerCallback):
    # def __init__(self, threshold):
    #     super().__init__()
    #     self.threshold = 0.25

    # def on_log(self, args, state, control, logs=None, **kwargs):
    #     current_wrr = logs.get('wrr')
    #     if current_wrr is not None and current_wrr <= self.threshold:
    #         logging.info(f"WRR 达到阈值 {current_wrr:.2%}，停止训练并保存模型。")
    #         control.should_save = True
    #         control.should_training_stop = True
    def __init__(self, thresholds, save_dir):
        super().__init__()
        self.thresholds = sorted(thresholds, reverse=True)  # 确保阈值按降序排列
        self.save_dir = save_dir
        self.saved_thresholds = set()

    def on_log(self, args, state, control, logs=None, **kwargs):
        current_wrr = logs.get('wrr')
        if current_wrr is not None:
            for threshold in self.thresholds:
                if current_wrr <= threshold and threshold not in self.saved_thresholds:
                    logging.info(f"WRR 达到阈值 {current_wrr:.2%}，保存模型。")
                    control.should_save = True
                    control.should_training_stop = False  # 不停止训练
                    self.saved_thresholds.add(threshold)
                    # 创建保存目录
                    save_path = os.path.join(self.save_dir, f"model_wrr_{threshold:.2f}")
                    os.makedirs(save_path, exist_ok=True)
                    # 保存模型为 pytorch_model.bin
                    model_save_path = os.path.join(save_path, "pytorch_model.bin")
                    torch.save(kwargs['model'].state_dict(), model_save_path)
                    logging.info(f"模型已保存到 {model_save_path}")
                    # 进行评估
                    # eval_results = kwargs['model']  
                    # logging.info(f"==========[{current_wrr} WRR TEST]=============")
                    # logging.info(f"+ Module CE Loss: {eval_results['eval_loss']:.4f}")
                    # logging.info(f"+ Module Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
                    break

def preprocess(raw_dataset, tokenizer):
    def _filter(examples):
        # filtered_data = []
        filtered_inputs = []
        filtered_labels = []
        for func_name, code_tokens, doc_tokens in zip(examples['func_name'], examples['func_code_tokens'],
                                                      examples['func_documentation_tokens']):
            if 'test' in func_name:  # CodeBERT: "function names with substring “test” are removed."
                continue

            if len(code_tokens) < 20:  # CodeBERT: "functions shorter than three lines are removed,"
                continue

            if len(doc_tokens) < 3:  # CodeBERT: " documentations shorter than three tokens are removed,":
                continue
            if not code_tokens or not doc_tokens:
                continue  # 跳过空的 code_tokens 或 doc_tokens

            input_text = ' '.join(code_tokens)
            label_text = ' '.join(doc_tokens)
            if not input_text.strip() or not label_text.strip():
                continue  # 跳过空字符串
            # filtered_data.append(' '.join(code_tokens + doc_tokens))
            filtered_inputs.append(' '.join(code_tokens))
            filtered_labels.append(' '.join(doc_tokens))
        results = {'inputs': filtered_inputs, 'labels': filtered_labels}
        # logging.info(f'filtered_data: {len(results["labels"]),results["labels"][0]}')
        # results = {'filtered_data': filtered_data}
        return results

    # def _tokenize_and_label(examples):
    #     tokenized_data = tokenizer(examples['filtered_data'], truncation=True, padding=True)
    #     # tokenized_data['labels'] = tokenized_data['input_ids'].copy()
    #     return tokenized_data
    def _tokenize_and_label(examples):
        model_inputs = tokenizer(examples['inputs'], truncation=True, padding='max_length', max_length=512)
        labels = tokenizer(text_target=examples['labels'], truncation=True, padding='max_length', max_length=128)

        model_inputs['labels'] = labels['input_ids']
        # logging.info(f'model_inputs: {model_inputs["labels"]}')
        return model_inputs
    
    filtered_data = raw_dataset.map(_filter, batched=True, num_proc=8,
                                    remove_columns=raw_dataset["train"].column_names, desc='filtering')
    preprocessed_data = filtered_data.map(_tokenize_and_label, batched=True, num_proc=8,
                                          remove_columns=filtered_data['train'].column_names, desc='tokenizing')
    return preprocessed_data

# def preprocess(raw_dataset, tokenizer):
#     def _filter(examples):
#         filtered_data = []
#         for func_name, code_tokens, doc_tokens in zip(examples['func_name'], examples['func_code_tokens'],
#                                                       examples['func_documentation_tokens']):
#             if 'test' in func_name:  # CodeBERT: "function names with substring “test” are removed."
#                 continue

#             if len(code_tokens) < 20:  # CodeBERT: "functions shorter than three lines are removed,"
#                 continue

#             if len(doc_tokens) < 3:  # CodeBERT: " documentations shorter than three tokens are removed,":
#                 continue

#             filtered_data.append(' '.join(code_tokens + doc_tokens))
#         results = {'filtered_data': filtered_data}
#         return results

#     def _tokenize_and_label(examples):
#         tokenized_data = tokenizer(examples['filtered_data'], truncation=True, padding=True)
#         # tokenized_data['labels'] = tokenized_data['input_ids'].copy()
#         return tokenized_data

#     filtered_data = raw_dataset.map(_filter, batched=True, num_proc=8,
#                                     remove_columns=raw_dataset["train"].column_names, desc='filtering')
#     preprocessed_data = filtered_data.map(_tokenize_and_label, batched=True, num_proc=8,
#                                           remove_columns=filtered_data['train'].column_names, desc='tokenizing')
#     return preprocessed_data

class Modularizer(Seq2SeqTrainer):
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
        self.log({'wrr': self.wrr.item()})
        # logging.info(f'loss_wrr: {loss_wrr:.2%},loss:{loss}')
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
            logging.info(f'\nWarning: module {module_val_mlm_loss} > model {self.model_val_mlm_loss} \n')
            # control.should_training_stop = True  # TEST
    def on_step_end(self, args, state, control, **kwargs):
        threshold = 0.25
        if self.wrr <= threshold:
            logging.info(f"Stopping training as wr={self.wrr} <= {threshold}")
            control.should_training_stop = True

def main():
    # model_path = f'./data/pretrain_model/codebert-base-mlm'
    # model_path = f'./data/pretrain_model/codet5_base'
    model_path = '/home/LAB/longwr/new_SeaM/TransModular_CodeT5/data/pretrain_model/codet5_small/'
    # if not os.path.exists(model_path):
    #     model_path = 'microsoft/codebert-base-mlm'
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id


    logging.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    logging.info(f"Model config vocab size: {model.config.vocab_size}")
    model = model.to('cuda')
    # model = RobertaForMaskedLM.from_pretrained(model_path)
    module = init_mask_model(model=model, no_mask=['lm_head'], is_low_rank=arguments.low_rank, rank=arguments.rank)
    module = module.to('cuda')
    logging.info(f'\n\n{module}\n\n')
    # tokenizer = RobertaTokenizer.from_pretrained(model_path)
    tokenizer.do_lower_case = True

    dataset_path = f'/home/LAB/longwr/new_SeaM/Tran_SeaM/data/dataset/code_search_net/dataset/{arguments.lang}'
    processed_dataset_path = f'{dataset_path}_processed_for_t5'
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
    # 在预处理后调用
    if not check_dataset(preprocessed_dataset):
        logging.info("Dataset contains None or empty values.")
        # 根据需要处理

    logging.info(f"Train dataset size:{len(preprocessed_dataset['train'])}" )
    logging.info(f"Eval dataset size:{len(preprocessed_dataset['test'])}")
    logging.info(f"Train dataset columns:{preprocessed_dataset['train'][0]}")
    # return 0
    # 检查是否有空样本
    for dataset_split in ['train', 'test']:
        for idx, sample in enumerate(preprocessed_dataset[dataset_split]):
            if 'input_ids' not in sample or 'labels' not in sample:
                logging.info(f"Sample {idx} in {dataset_split} split is missing 'input_ids' or 'labels'.")
                break
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model,padding='max_length',return_tensors='pt')
    mask_names = ['weight_mask', 'bias_mask']
    mask_param_names = [n for n, p in module.named_parameters() if any(mn in n for mn in mask_names)]
    logging.info('NOTE: this is a new version. lm_head is trained during modularizing.')
    lm_head_param_names = [n for n, p in module.lm_head.named_parameters()]

    logging.info(f'Trainable Parameters:')
    logging.info('='*50)
    logging.info(mask_param_names)
    logging.info('-'*50)
    logging.info(lm_head_param_names, "\n")

    mask_params = [p for n, p in module.named_parameters() if any(mn in n for mn in mask_names)]
    # NOTE: cannot directly find the keyword "lm_head" in module.named_parameters(), as the decoder in lm_head is not returned by module.named_parameters(). 
    # here need to use module.lm_head.named_parameters()
    lm_head_params = [p for n, p in module.lm_head.named_parameters()]

    # optimizer = AdamW(mask_params, lr=arguments.lr, weight_decay=arguments.weight_decay)
    optimizer = AdamW(
        [{'params': mask_params, 'lr': arguments.lr},
         {'params': lm_head_params, 'lr': arguments.lr_lmhead}], 
        weight_decay=arguments.weight_decay)

    module_save_dir = f'data/module_t5_small_{arguments.lang}/lr_{arguments.lr}_alpha_{arguments.alpha}_ne_{arguments.n_epochs}'
    if arguments.low_rank:
        module_save_dir = f'{module_save_dir}_rank_{arguments.rank}_lrlm_{arguments.lr_lmhead}'
    if arguments.early_stop:
        module_save_dir = f'{module_save_dir}_early_{arguments.eval_steps}'
    if arguments.warmup_ratio != 0.1:
        module_save_dir = f'{module_save_dir}_warm_{arguments.warmup_ratio}'

    training_args = Seq2SeqTrainingArguments(
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
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=preprocessed_dataset['train'],
            eval_dataset=preprocessed_dataset['test'],
            data_collator=data_collator,
        )
        model_eval_results = trainer.evaluate()
        logging.info(f'Model MLM Loss: {model_eval_results}\n\n')

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
            # callbacks=[StopOnWRRCallback(threshold=0.25)]
            callbacks=[StopOnWRRCallback(thresholds=[0.75, 0.5, 0.25,0.1], save_dir=module_save_dir)]
        )

        modularizer.train()
        modularizer.save_model(f'{module_save_dir}/result_{modularizer.wrr:.2}')

        logging.info('=' * 100)
        logging.info(f'WRR: {modularizer.wrr:.2%}')
        module_eval_results = modularizer.evaluate()
        # Since the masked tokens could be different in each step, the eval_results could be different in every evaluation.
        logging.info(f'Module MLM Loss: {module_eval_results}\n')
        logging.info(f'Model  MLM Loss: {model_eval_results}\n\n')
        torch.save(modularizer.model.state_dict(), f'{module_save_dir}/result_{modularizer.wrr:.2}/pytorch_model_try.bin')
        torch.save(module.state_dict(), f'{module_save_dir}/result_{modularizer.wrr:.2}/pytorch_model.bin')
    # elif arguments.do_eval:  # TODO: fix. TO load the checkpoint, i.e., the resulting module.
    #     module_eval_results = modularizer.evaluate()
    #     logging.info(f'Module: {module_eval_results}\n')


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
    parser.add_argument('--lr_lmhead', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--early_stop', action='store_true',
                        help='whether to early stop modularization '
                             'when the validation mlm loss of module is lower than that of model')
    parser.add_argument('--eval_steps', type=int, default=100)
    arguments = parser.parse_args()
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"./log/modularizer_t5_small_{arguments.lang}_lr_{arguments.lr}_alpha_{arguments.alpha}_ne_{arguments.n_epochs}_{current_time}.log"
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_filename),
                        logging.StreamHandler(sys.stdout)
                    ])
    logging.info(arguments)
    logging.info(arguments)

    tensorboard_dir = f'./tensorboard_log/modular_t5_small/{arguments.lang}/lr_{arguments.lr}_alpha_{arguments.alpha}_ne_{arguments.n_epochs}'
    if arguments.low_rank:
        assert arguments.rank > 0
        tensorboard_dir = f'{tensorboard_dir}_rank_{arguments.rank}_lrlm_{arguments.lr_lmhead}'
    if arguments.early_stop:
        tensorboard_dir = f'{tensorboard_dir}_early_{arguments.eval_steps}'
    if arguments.warmup_ratio != 0.1:
        tensorboard_dir = f'{tensorboard_dir}_warm_{arguments.warmup_ratio}'

    logging.info(f'tensorboard dir: {tensorboard_dir}')
    writer = SummaryWriter(tensorboard_dir)
    main()
