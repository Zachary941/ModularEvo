import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
# Add project path
sys.path.append('../../')
sys.path.append('../')
# Import model-related modules
from transformers import (GPTNeoForCausalLM, GPT2Tokenizer, 
                          AdamW, get_linear_schedule_with_warmup, Trainer, TrainingArguments)

from merge_methods.merging_methods import MergingMethod

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
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
                    logger.info(f"\n Early stop, best acc: {self.best_accuracy:.4f}")
    
            return metrics
            
    def training_step(self, model, inputs):
        if self.should_stop:
            logger.info(f"\n Early stop, best_accuracy: {self.best_accuracy:.4f}")
            logger.info(f"model save_path: {self.best_model_path}")
            raise RuntimeError("Early stopping triggered")
        
        return super().training_step(model, inputs)
        
    def _save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        save_path = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), save_path)
        logger.info(f"model be saved: {save_path}")
        
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


def merge_models(args, models_to_merge, stage ,base_model):
    """Merge models"""
    logger.info(f"Start stage {stage+1} model merging")
    
    merging_method = MergingMethod(args.merge_method)

    # Perform model merging
    scaling_coefficients = [args.alpha1, args.alpha2]
    merged_params = merging_method.get_merged_model(
        merged_model=base_model,
        models_to_merge=models_to_merge,
        exclude_param_names_regex=[".*classification.*"],
        scaling_coefficients=scaling_coefficients,
        models_use_deepcopy=True
    )
    
    # Save merged model parameters
    output_path = os.path.join(
        args.output_dir, 
        f"merged_stage{stage+1}_{args.merge_method}.bin"
    )
    torch.save(merged_params, output_path)
    logger.info(f"Merged model saved to {output_path}")
    
    return merged_params

