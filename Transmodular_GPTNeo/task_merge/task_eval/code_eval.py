import os
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset, load_from_disk, DatasetDict
import argparse
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, List
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

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
        logger.info(f"\n new acc: {current_accuracy:.4f}")

        return metrics
    def training_step(self, model, inputs):
        if self.should_stop:
            logger.info(f"\n Early stop,best_accuracy: {self.best_accuracy:.4f}")
            logger.info(f"model save_path: {self.best_model_path}")
            raise RuntimeError("Early stopping triggered")
        
        return super().training_step(model, inputs)
    def _save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        save_path = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), save_path)
        logger.info(f"model be saved:{save_path}")
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

# Data preprocessing function for code language identification
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
            raise ValueError("Need label_mapping")
            
        labels = [label_mapping[lang] for lang in examples['language_name']]
        inputs['labels'] = labels
        expected_length = len(examples['code'])
        for key, value in inputs.items():
            if len(value) != expected_length:
                raise ValueError(f"Incorrect: {key} length {len(value)}, expected_length {expected_length}")
        
        return inputs
        
    except Exception as e:
        logger.info(f"Error: {str(e)}")
        logger.info(f"Example code: {examples['code'][:1]}")
        logger.info(f"Label: {examples['language_name'][:1]}")
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
        logger.info(f"Error in compute_metrics: {str(e)}")
        logger.info(f"Label type: {type(labels)}, shape: {labels.shape if hasattr(labels, 'shape') else len(labels)}")
        logger.info(f"Prediction type: {type(preds)}, shape: {preds.shape if hasattr(preds, 'shape') else len(preds)}")
        raise

def evaluate_code_model(model: GPTNeoWithClassificationHead) -> dict:
    """
    Evaluate model on the Code language identification test set
    
    Args:
        model: The model to evaluate
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    try:
        # Load dataset
        LOCAL_DATASET_PATH = 'TransModular_GPT/fintune/data/code/'
        sampled_dataset_path = os.path.join(LOCAL_DATASET_PATH, "sampled_dataset")
        
        # Try to load from saved sampled dataset first
        if os.path.exists(sampled_dataset_path):
            logger.info(f"Loading sampled dataset from {sampled_dataset_path}")
            dataset = load_from_disk(sampled_dataset_path)
        else:
            # If not available, load and resplit the full dataset
            logger.info("Sampled dataset not found, loading full dataset and resampling")
            full_dataset = load_dataset(
                'parquet',  
                data_files={
                    'train': os.path.join(LOCAL_DATASET_PATH, "train-00000-of-00001-8b4da49264116bbf.parquet")
                }
            )
            
            # Extract 20,000 samples from the full dataset
            total_samples = 20000
            if len(full_dataset['train']) > total_samples:
                shuffled_dataset = full_dataset['train'].shuffle(seed=42)
                limited_dataset = shuffled_dataset.select(range(total_samples))
            else:
                limited_dataset = full_dataset['train']
            
            # Create split
            test_size = 6000  # 30% of 20,000
            val_size = 2000   # 10% of 20,000
            
            first_split = limited_dataset.train_test_split(test_size=test_size, seed=42)
            train_val = first_split["train"]
            test_dataset = first_split["test"]
            
            second_split = train_val.train_test_split(test_size=val_size, seed=42)
            train_dataset = second_split["train"]
            val_dataset = second_split["test"]
            
            dataset = DatasetDict({
                'train': train_dataset,
                'validation': val_dataset,
                'test': test_dataset
            })

        # Load tokenizer
        LOCAL_MODEL_PATH = 'TransModular_GPT/data/gpt-neo-125m/'
        tokenizer = GPT2Tokenizer.from_pretrained(LOCAL_MODEL_PATH)
        tokenizer.pad_token = tokenizer.eos_token

        # Process dataset
        label_mapping = create_label_mapping(dataset)
        tokenized_dataset = dataset.map(
            lambda x: preprocess_data(x, tokenizer, label_mapping=label_mapping),
            batched=True,
            remove_columns=dataset["test"].column_names
        )

        # Prepare model for evaluation
        model.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        print(f"label: {len(label_mapping)}")
        # Verify model's expected num_classes matches dataset's
        if model.num_classes != len(label_mapping):
            logger.warning(f"Model's num_classes ({model.num_classes}) doesn't match dataset labels ({len(label_mapping)})")

        # Setup evaluator
        eval_args = TrainingArguments(
            output_dir="eval_temp",
            per_device_eval_batch_size=8,
            remove_unused_columns=False,
        )

        evaluator = CustomTrainer(
            model=model,
            args=eval_args,
            compute_metrics=compute_metrics,
        )

        # Run evaluation
        results = evaluator.evaluate(tokenized_dataset['test'])
        
        # Extract metrics
        metrics = {
            'accuracy': results['eval_accuracy'],
            'micro_f1': results['eval_micro-f1'],
            'macro_f1': results['eval_macro-f1']
        }
        
        logger.info(f"Code language identification evaluation results: {metrics}")
        return metrics

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None