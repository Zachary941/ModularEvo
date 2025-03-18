from ast import arg
import copy
import os
import torch
import time
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset, DatasetDict
import argparse
import pandas as pd
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, List
from pathlib import Path
import logging
import sys
from new_optimizer import AdamWS
import random
import numpy as np

def load_langid_dataset(train_path, test_path):
    """Load LangID dataset directly from text files using the provided logic"""
    logger.info(f"Loading text files: {train_path} and {test_path}")
    
    def generate_examples(filepath):
        """Process a single file and generate examples"""
        logger.info(f"Generating examples from = {filepath}")
        examples = []
        try:
            with open(filepath, encoding="utf-8") as f:
                guid = 0
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Remove dataset prefixes
                    if "dataset10000, " in line:
                        line = line.replace('dataset10000, ', '')
                    elif "dataset50000, " in line:
                        line = line.replace('dataset50000, ', '')
                    
                    # Create instance
                    instance = {
                        "id": str(guid),
                        "language": line[-2:],
                        "sentence": line[:-3],
                    }
                    
                    examples.append(instance)
                    guid += 1
            
            logger.info(f"Generated {len(examples)} examples from {filepath}")
            return examples
            
        except Exception as e:
            logger.error(f"Error processing file {filepath}: {str(e)}")
            raise
    
    # Generate examples for train and test files
    train_examples = generate_examples(train_path)
    test_examples = generate_examples(test_path)

    save_dir = os.path.dirname('./')
    save_path = os.path.join(save_dir, "train_examples.txt")
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f"Total examples: {len(train_examples)}\n\n")
        for example in train_examples:
            f.write(f"ID: {example['id']}\n")
            f.write(f"Language: {example['language']}\n")
            f.write("="*50 + "\n")
    logger.info(f"Saved train examples to: {save_path}")
    
    # Create datasets
    train_dataset = Dataset.from_list(train_examples)
    test_dataset = Dataset.from_list(test_examples)
    
    logger.info(f"Successfully loaded training set ({len(train_dataset)} items) and test set ({len(test_dataset)} items)")
    
    return {
        "train": train_dataset,
        "test": test_dataset
    }

def process_langid(examples):
    """Process LangID text, extract sentences and language labels"""
    processed = {
        "id": [str(i) for i in range(len(examples["sentence"]))],
        "sentence": [],
        "language": []
    }
    
    for text in examples["sentence"]:
        text = str(text).strip()  
        if "dataset10000, " in text:
            text = text.replace("dataset10000, ", "")
        elif "dataset50000, " in text:
            text = text.replace("dataset50000, ", "")
            
        if len(text) >= 3:
            processed["sentence"].append(text[:-3])
            processed["language"].append(text[-2:])
        else:
            logger.warning(f"Abnormal text format: '{text}'")
    
    return processed
def create_label_mapping(dataset) -> Dict[str, int]:

    unique_topics = set()
    for split in dataset.keys():
        unique_topics.update(set(dataset[split]['language']))
    return {topic: idx for idx, topic in enumerate(sorted(unique_topics))}

def prepare_langid_dataset(args):
    """Prepare LangID dataset, split training set into train and validation sets"""
    tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    train_path = os.path.join(args.data_path, f"nordic_dsl_10000train.csv")
    test_path = os.path.join(args.data_path, f"nordic_dsl_10000test.csv")
    
    processed_dataset = load_langid_dataset(train_path, test_path)
    
    # Split training set into train and validation sets
    train_val_split = processed_dataset["train"].train_test_split(
        test_size=0.1,  # 10% as validation set
        seed=args.seed
    )
    
    # Create new dataset dictionary
    final_datasets = DatasetDict({
        "train": train_val_split["train"],
        "validation": train_val_split["test"],
        "test": processed_dataset["test"]
    })
    
    logger.info(f"Dataset sizes after splitting:")
    logger.info(f"  Train: {len(final_datasets['train'])} samples")
    logger.info(f"  Validation: {len(final_datasets['validation'])} samples")
    logger.info(f"  Test: {len(final_datasets['test'])} samples")
    
    label_mapping = create_label_mapping(final_datasets)
    num_labels = len(label_mapping)
    logger.info(f"label num: {num_labels}")
    logger.info("mapping:", label_mapping)

    # Tokenize text
    def tokenize_function(examples, tokenizer, label_mapping=None, max_length=512):
        inputs = tokenizer(
            examples['sentence'],
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors=None  
        )
        labels = [label_mapping[topic] for topic in examples['language']]
        inputs['labels'] = labels
        expected_length = len(examples['sentence'])
        for key, value in inputs.items():
            if len(value) != expected_length:
                raise ValueError(f"incorrte: {key} length {len(value)},expected_length {expected_length}")
        return inputs
    # Apply tokenization to all datasets
    tokenized_datasets = DatasetDict()
    tokenized_datasets = final_datasets.map(
        lambda x: tokenize_function(x, tokenizer,label_mapping=label_mapping),
        batched=True,
        remove_columns=final_datasets["train"].column_names
    )
    logger.info("\dataset:")
    for split in tokenized_datasets.keys():  
        logger.info(f"\n{split} :")
        logger.info(f"sample num: {len(tokenized_datasets[split])}")
        logger.info(f"label : {torch.bincount(torch.tensor(tokenized_datasets[split]['labels']))}")

    for split, dataset in tokenized_datasets.items():
        logger.info(f"{split} set: {len(dataset)} samples")
        if len(dataset) > 0:
            logger.info(f"{split} sample example: {dataset[0]}")
    
    return tokenized_datasets, tokenizer, num_labels

# Define GPT-Neo model with classification head
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

# Custom trainer with early stopping support
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
                logger.info(f"\nNew best accuracy: {current_accuracy:.4f}")
                logger.info(f"Best model saved to: {save_path}")
            else:
                self.no_improve_count += 1
                logger.info(f"No improvement for {self.no_improve_count} consecutive steps")
                
                if self.no_improve_count >= self.patience:
                    self.should_stop = True
                    logger.info(f"\nEarly stopping condition reached, best accuracy: {self.best_accuracy:.4f}")
    
        return metrics

    def training_step(self, model, inputs):
        if self.should_stop:
            logger.info(f"\nEarly stopping training, best accuracy: {self.best_accuracy:.4f}")
            logger.info(f"Best model path: {self.best_model_path}")
            raise RuntimeError("Early stopping triggered")
        
        return super().training_step(model, inputs)

    def _save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        save_path = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), save_path)
        logger.info(f"Model saved: {save_path}")
        
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        labels = inputs['labels']

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss']
        logits = outputs['logits']
        return (loss, logits) if return_outputs else loss

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

# Calculate evaluation metrics
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
        logger.error(f"Error computing metrics: {str(e)}")
        logger.info(f"Labels type: {type(labels)}, shape: {labels.shape if hasattr(labels, 'shape') else len(labels)}")
        logger.info(f"Predictions type: {type(preds)}, shape: {preds.shape if hasattr(preds, 'shape') else len(preds)}")
        raise
def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def main(args):
    """Main function for training and evaluating the model"""
    # Set random seed
    set_seed(args.seed)
    
    # Prepare dataset
    tokenized_datasets, tokenizer, num_labels = prepare_langid_dataset(args)
    
    # Initialize model
    model = GPTNeoWithClassificationHead(args.pretrained_model_path, num_classes=num_labels)
    logger.info(f"Model initialized with {num_labels} classes")
    baes_model = copy.deepcopy(model)
    base_state_dict = baes_model.state_dict()
    # If using mask training
    module_state = None
    if args.use_mask:
        if args.mask_rate == 0.25:
            mask_path = "TransModular_GPT/data/module_europarl/lr_0.005_alpha_10.0_bs_4_time_20250306_205212/model_wrr_0.25/pytorch_model.bin"

        logger.info(f"Loading mask: {mask_path}")
        module_state = torch.load(mask_path)
        
        # Count mask parameters
        masked_params_count = 0
        total_params_count = 0
        
        for name, param in model.named_parameters():
            if "classification" not in name:
                modify_name = name.replace("base_model.", "")
                if f"{modify_name}_mask" in module_state:
                    total_params_count += param.numel()
                    mask = module_state[f'{modify_name}_mask']
                    bin_mask = (mask > 0).float()
                    masked_params_count += bin_mask.sum().item()
        
        logger.info(f"Total parameters: {total_params_count}")
        logger.info(f"Parameters in mask: {masked_params_count} ({masked_params_count/total_params_count:.2%})")
    
    # Create optimizer
    optimizer_kwargs = {
        "lr": args.lr,
        "weight_decay": 0
    }
    optimizer = AdamWS(
        [model.named_parameters()],
        mask_dict=module_state,
        **optimizer_kwargs
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        save_steps=500,
        save_total_limit=2,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=100,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=100,
        save_strategy="steps",
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        dataloader_drop_last=False,
        remove_unused_columns=False,
        prediction_loss_only=False,
        eval_accumulation_steps=None,
        save_safetensors=False,
        push_to_hub=False,
    )
    
    # Initialize trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        optimizers=(optimizer, None),
        compute_metrics=compute_metrics,
        patience=args.patience,
        min_delta=args.min_delta,
    )
    
    # Start training
    logger.info("Starting training...")
    try:
        trainer.train()
    except RuntimeError as e:
        if "Early stopping triggered" in str(e):
            logger.info("Early stopping training...")
        else:
            raise
    
    # Evaluate best model
    if trainer.best_model_path is not None:
        logger.info(f"\nTraining completed")
        logger.info(f"Best accuracy: {trainer.best_accuracy:.4f}")
        logger.info(f"Best model path: {trainer.best_model_path}")
        
        logger.info("\nStarting testing...")
        best_model = GPTNeoWithClassificationHead(args.pretrained_model_path, num_classes=num_labels)
        checkpoint = torch.load(trainer.best_model_path)
        best_model.load_state_dict(checkpoint)
        
        best_trainer = CustomTrainer(
            model=best_model,
            args=training_args,
            eval_dataset=tokenized_datasets["test"],
            compute_metrics=compute_metrics,
        )
        
        test_results = best_trainer.evaluate(metric_key_prefix="test")
        logger.info(f"Test results: {test_results}")
    
    # Save final model
    save_dir = Path(args.output_dir) / 'result'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "pytorch_model.bin"
    torch.save(model.state_dict(), save_path)
    logger.info(f"Final model saved to: {save_path}")
    if args.use_mask:
        total_params_count = 0
        change_params_count = 0

        for name, param in best_model.named_parameters():
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

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fine-tune GPT-Neo 125M on Nordic Language ID dataset.")
    parser.add_argument('--lr', type=float, default=5e-5,
                      help="Learning rate (default: 5e-5)")
    parser.add_argument('--weight_decay', type=float, default=0.01,
                      help="Weight decay (default: 0.01)")
    parser.add_argument('--batch_size', type=int, default=8,
                      help="Training batch size")
    parser.add_argument('--epochs', type=int, default=3,
                      help="Number of training epochs (default: 3)")
    parser.add_argument('--patience', type=int, default=3,
                      help="Early stopping patience (default: 3)")
    parser.add_argument('--min_delta', type=float, default=0.001,
                      help="Minimum improvement threshold (default: 0.001)")
    parser.add_argument('--use_mask', action='store_true', 
                      help="Use mask for fine-tuning")
    parser.add_argument('--mask_rate', type=float, default=0.25,
                      help="Mask rate (default: 0.25)")
    parser.add_argument('--max_length', type=int, default=512,
                      help="Maximum sequence length (default: 512)")
    parser.add_argument('--data_path', type=str, default="./data/langid",
                      help="Dataset path (default: ./data/langid)")
    parser.add_argument('--pretrained_model_path', type=str, 
                      default="TransModular_GPT/data/gpt-neo-125m/",
                      help="Pre-trained model path")
    parser.add_argument('--seed', type=int, default=42,
                      help="Random seed (default: 42)")
    parser.add_argument('--output_dir', type=str, default=None,
                      help="Output directory (default: auto-generated)")
    
    args = parser.parse_args()
    
    # Set output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        if args.use_mask:
            base_output_dir = f'TransModular_GPT/finetune/save_model_with_mask_{args.mask_rate}/'
        else:
            base_output_dir = 'TransModular_GPT/finetune/save_model/'
        
        args.output_dir = os.path.join(
            base_output_dir, 
            "langid", 
            f"{timestamp}_lr{args.lr}_bs{args.batch_size}_e{args.epochs}_p{args.patience}"
        )
    
    os.makedirs(args.output_dir, exist_ok=True)
    log_file_path = os.path.join(args.output_dir, "output.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
        ]
    )
    logger = logging.getLogger(__name__)
    
    # logger.info parameters
    logger.info(f"Running parameters: {vars(args)}")
    
    # Call main function
    main(args)