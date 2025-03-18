import os
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import argparse
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from sklearn.metrics import accuracy_score,f1_score
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
            
            # logger.info(f"Debug - Batch sizes: labels={labels.shape}, logits={logits.shape}")
            
            return (loss, logits, labels)
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        labels = inputs['labels']

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask ,labels=labels)
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

# Data preprocessing function
def create_label_mapping(dataset) -> Dict[str, int]:

    unique_topics = set()
    for split in dataset.keys():
        unique_topics.update(set(dataset[split]['topic']))
    return {topic: idx for idx, topic in enumerate(sorted(unique_topics))}

def preprocess_data(examples, tokenizer, label_mapping=None, max_length=512):
    try:
        inputs = tokenizer(
            examples['question'],
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors=None  
        )
        
        if label_mapping is None:
            raise ValueError("need label_mapping")
            
        labels = [label_mapping[topic] for topic in examples['topic']]
        inputs['labels'] = labels
        expected_length = len(examples['question'])
        for key, value in inputs.items():
            if len(value) != expected_length:
                raise ValueError(f"incorrte: {key} length {len(value)},expected_length {expected_length}")
        
        
        return inputs
        
    except Exception as e:
        logger.info(f"error: {str(e)}")
        logger.info(f"example question: {examples['question'][:1]}")
        logger.info(f"label: {examples['topic'][:1]}")
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
        logger.info(f"pre type: {type(preds)}, shape: {preds.shape if hasattr(preds, 'shape') else len(preds)}")
        raise

def evaluate_math_model(model: GPTNeoWithClassificationHead) -> dict:
    """
    Evaluate model on the Math-QA test set
    
    Args:
        model: The model to evaluate
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    try:
        # Load dataset
        LOCAL_DATASET_PATH = 'TransModular_GPT/finetune/data/mathqa/' 
        dataset = load_dataset(
            'parquet',  
            data_files={
                'train': os.path.join(LOCAL_DATASET_PATH, "train-00000-of-00001.parquet"),
                'validation': os.path.join(LOCAL_DATASET_PATH, "val-00000-of-00001.parquet"),
                'test': os.path.join(LOCAL_DATASET_PATH, "test-00000-of-00001.parquet")
            }
        )

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
        
        return metrics

    except Exception as e:
        logger.info(f"Evaluation failed: {str(e)}")
        return None