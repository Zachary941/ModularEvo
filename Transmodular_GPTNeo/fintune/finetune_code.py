from ast import arg
import copy
import os
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset,DatasetDict
import argparse
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, List
from pathlib import Path
import logging
import sys
from new_optimizer import AdamWS


# Define a new model with a classification head
class GPTNeoWithClassificationHead(nn.Module):
    def __init__(self, base_model_name, num_classes):
        super().__init__()
        self.base_model = GPTNeoForCausalLM.from_pretrained(base_model_name)
        self.hidden_size = self.base_model.config.hidden_size
        self.num_classes = num_classes
        self.base_model.config.output_hidden_states = True
        self.classification_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 2, num_classes)
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.base_model(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        
        last_hidden_state = outputs.hidden_states[-1]
        sentence_representation = torch.mean(last_hidden_state, dim=1)  
        layer_norm = nn.LayerNorm(self.hidden_size).to(last_hidden_state.device)
        sentence_representation = layer_norm(sentence_representation)
        
        logits = self.classification_head(sentence_representation)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            
        return {'loss': loss, 'logits': logits} if loss is not None else {'logits': logits}

# Custom Trainer class to apply mask during training
class CustomTrainer(Trainer):
    def __init__(self, *args, patience=3, min_delta=0.001, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_accuracy = 0.0
        self.best_model_path = None
        self.patience = patience  
        self.min_delta = min_delta 
        self.no_improve_count = 0
        self.should_stop = False

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )
        
        current_accuracy = metrics.get(f"{metric_key_prefix}_accuracy")
        if current_accuracy is not None:
            if current_accuracy > self.best_accuracy + self.min_delta:
                self.best_accuracy = current_accuracy
                self.no_improve_count = 0  
                
                save_dir = Path(self.args.output_dir) / 'best_model'
                save_dir.mkdir(parents=True, exist_ok=True)
                model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                save_path = save_dir / "pytorch_model.bin"
                
                torch.save(model_to_save.state_dict(), save_path)
                
                self.best_model_path = save_path
                logger.info(f"\nnew acc: {current_accuracy:.4f}")
                logger.info(f"save_path: {save_path}")
            else:
                self.no_improve_count += 1
                logger.info(f" {self.no_improve_count} step no improvement")
                
                if self.no_improve_count >= self.patience:
                    self.should_stop = True
                    logger.info(f"\nEarly stop, best acc: {self.best_accuracy:.4f}")
        
        return metrics
        
    def training_step(self, model, inputs):
        if self.should_stop:
            logger.info(f"\nEarly stop, best_accuracy: {self.best_accuracy:.4f}")
            logger.info(f"model save_path: {self.best_model_path}")
            raise RuntimeError("Early stopping triggered")
        
        return super().training_step(model, inputs)
        
    def _save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        save_path = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), save_path)
        logger.info(f"model saved: {save_path}")
        
    def _save_checkpoint(self, model, trial, metrics=None):
        checkpoint_folder = f"checkpoint-{self.state.global_step}"
        run_dir = os.path.join(self.args.output_dir, checkpoint_folder)
        os.makedirs(run_dir, exist_ok=True)
        return super()._save_checkpoint(model, trial, metrics)
        
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            inputs = self._prepare_inputs(inputs)
            if ignore_keys is None:
                ignore_keys = []

            outputs = model(**inputs)
            loss = outputs['loss'] if 'loss' in outputs else None
            logits = outputs['logits']
            labels = inputs['labels']
            return (loss, logits, labels)
            
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        labels = inputs['labels']

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss']
        logits = outputs['logits']
        return (loss, logits) if return_outputs else loss

def create_label_mapping(dataset) -> Dict[str, int]:
    """Create mapping from language names to indices"""
    unique_languages = set()
    for split in dataset.keys():
        unique_languages.update(set(dataset[split]['language_name']))
    return {lang: idx for idx, lang in enumerate(sorted(unique_languages))}

def preprocess_data(examples, tokenizer, label_mapping=None, max_length=512):
    """Preprocess code data examples"""
    try:
        inputs = tokenizer(
            examples['code'],
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors=None  
        )
        
        if label_mapping is None:
            raise ValueError("need label_mapping")
            
        labels = [label_mapping[lang] for lang in examples['language_name']]
        inputs['labels'] = labels
        expected_length = len(examples['code'])
        for key, value in inputs.items():
            if len(value) != expected_length:
                raise ValueError(f"incorrect: {key} length {len(value)}, expected_length {expected_length}")
        
        return inputs
        
    except Exception as e:
        logger.info(f"error: {str(e)}")
        logger.info(f"example code: {examples['code'][:1]}")
        logger.info(f"label: {examples['language_name'][:1]}")
        raise

# Compute evaluation metrics
def compute_metrics(pred):
    try:
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        acc = accuracy_score(labels, preds)
        micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)

        return {
            "accuracy": acc,
            "micro-f1": micro_f1,
            "macro-f1": macro_f1,
        }
    except Exception as e:
        logger.info(f"error in compute_metrics: {str(e)}")
        logger.info(f"label type: {type(labels)}, shape: {labels.shape if hasattr(labels, 'shape') else len(labels)}")
        logger.info(f"prediction type: {type(preds)}, shape: {preds.shape if hasattr(preds, 'shape') else len(preds)}")
        raise

def main(args):
    # Load Code dataset
    LOCAL_DATASET_PATH = 'TransModular_GPT/finetune/data/code/' 
    full_dataset = load_dataset(
        'parquet',  
        data_files={
            'train': os.path.join(LOCAL_DATASET_PATH, "train-00000-of-00001-8b4da49264116bbf.parquet")
        }
    )


    logger.info(f"Loaded original dataset with {len(full_dataset['train'])} samples")
    # Extract only 20,000 samples from the full dataset
    total_samples = 20000
    if len(full_dataset['train']) > total_samples:
        # Shuffle dataset and select first 20,000 samples
        shuffled_dataset = full_dataset['train'].shuffle(seed=42)
        limited_dataset = shuffled_dataset.select(range(total_samples))
        logger.info(f"Limited dataset to {len(limited_dataset)} samples")
    else:
        limited_dataset = full_dataset['train']
        logger.info(f"Using all {len(limited_dataset)} samples (less than {total_samples} available)")
    
    # Split into train (60%), validation (10%), test (30%)
    train_size = 12000  # 60% of 20,000
    val_size = 2000     # 10% of 20,000
    test_size = 6000    # 30% of 20,000
    
    # First split: 70% train+val, 30% test
    split_dataset = limited_dataset.train_test_split(test_size=test_size, seed=42)
    train_valid = split_dataset["train"]
    test_dataset = split_dataset["test"]
    
    # Second split: divide train+val into train and validation
    split_train_valid = train_valid.train_test_split(test_size=val_size, seed=42)
    train_dataset = split_train_valid["train"]
    valid_dataset = split_train_valid["test"]
    
    # Create a dataset dictionary with all splits

    dataset = DatasetDict({
        'train': train_dataset,
        'validation': valid_dataset,
        'test': test_dataset
    })
    
    logger.info(f"Dataset splits created:")
    logger.info(f"  - Train: {len(dataset['train'])} samples ({len(dataset['train'])/len(full_dataset['train']):.1%})")
    logger.info(f"  - Validation: {len(dataset['validation'])} samples ({len(dataset['validation'])/len(full_dataset['train']):.1%})")
    logger.info(f"  - Test: {len(dataset['test'])} samples ({len(dataset['test'])/len(full_dataset['train']):.1%})")

    sampled_dataset_dir = os.path.join(LOCAL_DATASET_PATH, "sampled_dataset")
    os.makedirs(sampled_dataset_dir, exist_ok=True)
    dataset.save_to_disk(sampled_dataset_dir)
    logger.info(f"Sampled dataset saved to {sampled_dataset_dir}")
    
    # Load model and tokenizer
    LOCAL_MODEL_PATH = 'TransModular_GPT/data/gpt-neo-125m/'
    tokenizer = GPT2Tokenizer.from_pretrained(LOCAL_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    # Preprocess the dataset
    label_mapping = create_label_mapping(dataset)
    num_labels = len(label_mapping)
    logger.info(f"label count: {num_labels}")
    logger.info(f"label mapping: {label_mapping}")
    tokenized_datasets = dataset.map(
        lambda x: preprocess_data(x, tokenizer, label_mapping=label_mapping),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    logger.info("\ndataset info:")
    for split in tokenized_datasets.keys():  
        logger.info(f"\n{split} dataset:")
        logger.info(f"sample count: {len(tokenized_datasets[split])}")
        logger.info(f"label distribution: {torch.bincount(torch.tensor(tokenized_datasets[split]['labels']))}")

    # Initialize model
    model = GPTNeoWithClassificationHead(LOCAL_MODEL_PATH, num_classes=num_labels)
    baes_model = copy.deepcopy(model)
    base_state_dict = baes_model.state_dict()
    
    if args.use_mask:
        if args.mask_rate == 0.25:
            module_state = torch.load("TransModular_GPT/data/module_github/model_wrr_0.25/pytorch_model.bin")
        elif args.mask_rate == 0.5:
            module_state = torch.load("TransModular_GPT/data/module_github/model_wrr_0.50/pytorch_model.bin")
        elif args.mask_rate == 0.75:
            module_state = torch.load("TransModular_GPT/data/module_github/model_wrr_0.75/pytorch_model.bin")
        for name, param in model.named_parameters():
            param.requires_grad = False
        masked_params_count = 0
        total_params_count = 0
        orig_params = {name: param.clone().detach() for name, param in model.named_parameters()}
        masked_params = []
        for name, param in model.named_parameters():
            modify_name = name.replace("base_model.", "")
            if f"{modify_name}_mask" in module_state:
                param.requires_grad = True
                masked_params.append(param)
                total_params_count += param.numel()
                mask = module_state[f'{modify_name}_mask']
                bin_mask = (mask > 0).float()
                masked_params_count += bin_mask.sum().item()
                if bin_mask.sum().item() > 0:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = False
        logger.info(f"Total parameters: {total_params_count}")
        logger.info(f"Parameters in mask: {masked_params_count} ({masked_params_count/total_params_count:.2%})")
        
        total_params_count = 0
        change_params_count = 0

        for name, param in model.named_parameters():
            modify_name = name.replace("base_model.", "")
            if f"{modify_name}_mask" in module_state:
                total_params_count += param.numel()
                orig_param = base_state_dict[name].to(param.device)
                changes = (param - orig_param).abs() > 1e-6
                change_params_count += changes.sum().item()
                change_ratio = changes.sum().item() / changes.numel()
        
        logger.info(f"Total parameters: {total_params_count}")
        logger.info(f"Changed parameters: {change_params_count} ({change_params_count/total_params_count:.2%})")

    # Create optimizer
    optimizer_kwargs = {
        "lr": args.lr,
        "weight_decay": 0
    }
    def filter_trainable_params():
        for name, param in model.named_parameters():
            if param.requires_grad:
                yield (name, param)
    if args.use_mask:
        optimizer = AdamWS(
            [filter_trainable_params()],
            mask_dict=module_state,
            **optimizer_kwargs
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    
    # Create output directory
    output_dir = args.output_dir

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=10000,
        save_total_limit=2,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        learning_rate=args.lr,
        warmup_steps=500,
        save_strategy="steps",
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        per_device_eval_batch_size=args.batch_size,  
        dataloader_drop_last=False,    
        remove_unused_columns=False,   
        prediction_loss_only=False,   
        eval_accumulation_steps=None,
        save_safetensors=False,  
        push_to_hub=False, 
    )

    # Initialize Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'], 
        tokenizer=tokenizer,
        optimizers=(optimizer, None),
        compute_metrics=compute_metrics,
        patience=args.patience, 
        min_delta=0.001, 
    )

    # Start training
    try:
        trainer.train()
    except RuntimeError as e:
        if "Early stopping triggered" in str(e):
            logger.info("early stop...")
        else:
            raise
            
    if args.use_mask:
        total_params_count = 0
        change_params_count = 0

        for name, param in model.named_parameters():
            modify_name = name.replace("base_model.", "")
            if f"{modify_name}_mask" in module_state:
                total_params_count += param.numel()
                orig_param = base_state_dict[name].to(param.device)
                changes = (param - orig_param).abs() > 1e-6
                change_params_count += changes.sum().item()
                change_ratio = changes.sum().item() / changes.numel()
                logger.info(f"layer {name} change ratio:{change_ratio:.2%}")
        
        logger.info(f"Total parameters: {total_params_count}")
        logger.info(f"Changed parameters: {change_params_count} ({change_params_count/total_params_count:.2%})")

    if trainer.best_model_path is not None:
        logger.info(f"\nfinished training")
        logger.info(f"best_accuracy: {trainer.best_accuracy:.4f}")
        logger.info(f"best_model_path: {trainer.best_model_path}")
        
        logger.info("\ntesting...")
        best_model = GPTNeoWithClassificationHead(LOCAL_MODEL_PATH, num_classes=num_labels)
        checkpoint = torch.load(trainer.best_model_path)
        best_model.load_state_dict(checkpoint)
        
        best_trainer = CustomTrainer(
            model=best_model,
            args=training_args,
            eval_dataset=tokenized_datasets['test'],
            compute_metrics=compute_metrics,
        )
        
        test_results = best_trainer.evaluate()
        logger.info(f"test_results: {test_results}")
        
    # Save the final model
    save_dir = Path(output_dir) / 'result'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "pytorch_model.bin"
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    # Parse command-line arguments

    parser = argparse.ArgumentParser(description="Fine-tune GPT-Neo 125M on code language identification dataset")
    parser.add_argument('--lr', type=float, default=5e-5,
                      help="Learning rate (default: 5e-5)")
    parser.add_argument('--batch_size', type=int, default=8,
                      help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=2,
                      help="Number of training epochs (default: 2)")
    parser.add_argument('--patience', type=int, default=3,
                      help="Early stopping patience (default: 3)")
    parser.add_argument('--min_delta', type=float, default=0.001,
                      help="Minimum change in accuracy to qualify as an improvement (default: 0.001)")
    parser.add_argument('--use_mask', action='store_true', 
                      help="Use mask to fine-tune only part of the model weights")
    parser.add_argument('--mask_rate', type=float, default=0.25,
                      help="Mask rate for masked fine-tuning")
    parser.add_argument('--output_dir', type=str, default="./data",
                      help="Directory to save the fine-tuned model (default: ./data)")
    
    args = parser.parse_args()
    if args.use_mask:
        base_output_dir = f'TransModular_GPT/finetune/save_model_with_mask_{args.mask_rate}/'
    else:
        base_output_dir = 'TransModular_GPT/finetune/save_model/'

    output_dir = os.path.join(base_output_dir, "code", f"lr{args.lr}_bs{args.batch_size}_e{args.epochs}_p{args.patience}")
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = Path(f"{output_dir}/output.log")
    args.output_dir = output_dir
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Call the main function
    main(args)