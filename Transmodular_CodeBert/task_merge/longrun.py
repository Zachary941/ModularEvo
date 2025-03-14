# -*- coding: utf-8 -*-
"""
Long-process training script:
1. Split the dataset into 3 parts
2. Fine-tune two task models on each part of the data respectively
3. Merge the two task models
4. Implement two fine-tuning strategies: full fine-tuning and masked fine-tuning
"""

import os
import sys
import argparse
import json
import random
import logging
import shutil
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, SequentialSampler, RandomSampler 
import multiprocessing
from tqdm import tqdm
# Add project path
sys.path.append('../../')
sys.path.append('../')

# Import model-related modules
from transformers import (RobertaModel, RobertaTokenizer, RobertaConfig,
                          AdamW, get_linear_schedule_with_warmup)
from task_eval.code_clone_eval import evaluate as evaluate_clone
from task_eval.code_clone_eval import Model as Model_clone, load_and_cache_examples
from task_eval.nl_code_search_eval import evaluate as evaluate_search
from task_eval.nl_code_search_eval import Model as Model_search, TextDataset

# Import merging-related modules 
from merge_methods.merging_methods import MergingMethod

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Set multi-process sharing strategy to file system
torch.multiprocessing.set_sharing_strategy('file_system')

def parse_args():
    """Parse command-line arguments """
    parser = argparse.ArgumentParser("Long-process training and model merging framework")
    # Basic settings
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--output_dir", type=str, default="./longrun_output", help="Output directory")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    
    # Fine-tuning strategy
    parser.add_argument("--tuning_strategy", type=str, default="full", 
                         choices=["full", "mask"], help="Fine-tuning strategy: full or mask")

    # Merging method
    parser.add_argument("--merge_method", type=str, default="task_arithmetic", 
                        choices=["average_merging", "task_arithmetic", "fisher_merging", "ties_merging"],
                        help="Merging method")
    parser.add_argument("--alpha1", type=float, default=0.5, help="Merging coefficient for task 1")
    parser.add_argument("--alpha2", type=float, default=0.5, help="Merging coefficient for task 2")
    
    # Path settings
    parser.add_argument("--pretrained_model_path", type=str, 
                        default="data/pretrain_model/codebert-base/",
                        help="Pre-trained model path")
    parser.add_argument("--code_clone_data_path", type=str, 
                        default="Clone_detection_BigCloneBench_2/dataset/",
                        help="Code Clone dataset path")
    parser.add_argument("--nl_code_search_data_path", type=str, 
                        default="NL_code_search_WebQuery/CoSQA/",
                        help="NL Code Search dataset path")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    return args

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def split_dataset(dataset, num_splits=3):
    """Split the dataset into multiple parts"""
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    
    split_size = dataset_size // num_splits
    splits = []
    
    for i in range(num_splits):
        if i < num_splits - 1:
            split_indices = indices[i * split_size:(i+1) * split_size]
        else:
            split_indices = indices[i * split_size:]
        splits.append(Subset(dataset, split_indices))
    
    return splits

def full_finetune(args, model, optimizer, train_dataloader, device, task_type):
    """Fully fine-tune the model"""
    model.train()
    model.to(device)
    total_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Full Fine-tuning ({task_type})", leave=True)
    
    for batch in progress_bar:
        # Process input
        if task_type == "code_clone":
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            loss, _ = model(inputs, labels)
        else:  # nl_code_search
            code_inputs = batch[0].to(device)
            nl_inputs = batch[1].to(device)
            labels = batch[2].to(device)
            outputs = model(code_inputs, nl_inputs, labels)
            loss = outputs[0] if isinstance(outputs, tuple) else outputs
        
        
        # Backpropagation
        loss.backward()
        total_loss += loss.item()

        progress_bar.set_postfix(loss=loss.item(), avg_loss=total_loss/(progress_bar.n+1))
        
        # Parameter update
        optimizer.step()
        optimizer.zero_grad()
    
    return total_loss / len(train_dataloader)

def mask_finetune(args, model, optimizer, train_dataloader, device, task_type, module_state):
    """Fine-tune using masks"""
    model.train()
    model.to(device)
    total_loss = 0
    module_state_dict = torch.load(module_state, map_location=device)
    true_elements = 0
    total_elements = 0
    for name, param in model.named_parameters():
        modified_name = name.replace('encoder.', 'roberta.', 1)
        if f'{modified_name}_mask' in module_state_dict:
            mask = module_state_dict[f'{modified_name}_mask']
            bin_mask = (mask > 0).float()
            true_elements += torch.sum(bin_mask)
            total_elements += bin_mask.numel()
    
    if total_elements > 0:
        logger.info(f"Mask sparsity: {100.0 * (1 - true_elements / total_elements):.2f}% "
                   f"({true_elements}/{total_elements} parameters will be updated)")
    

    progress_bar = tqdm(train_dataloader, desc=f"Masked Fine-tuning ({task_type})", leave=True)
    
    for batch in progress_bar:
        # Process input
        if task_type == "code_clone":
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            loss, _ = model(inputs, labels)
        else:  # nl_code_search
            code_inputs = batch[0].to(device)
            nl_inputs = batch[1].to(device)
            labels = batch[2].to(device)
            outputs = model(code_inputs, nl_inputs, labels)
            loss = outputs[0] if isinstance(outputs, tuple) else outputs
        
        
        # Backpropagation
        loss.backward()
        
        # Apply masks to gradients
        for name, param in model.named_parameters():
            modified_name = name.replace('encoder.', 'roberta.', 1)
            if f'{modified_name}_mask' in module_state_dict:
                mask = module_state_dict[f'{modified_name}_mask']
                bin_mask = (mask > 0).float().to(device)
                param.grad = param.grad * bin_mask

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item(), avg_loss=total_loss/(progress_bar.n+1))
        
        # Parameter update
        optimizer.step()
        optimizer.zero_grad()
    
    return total_loss / len(train_dataloader)

def train_model(args, model, train_dataloader,  task_type, stage):
    """Train the model"""
    model.to(args.device)
    
    # Set up optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5)
    
    logger.info(f"Start {task_type} task stage {stage+1} training")
    
    best_metric = 0
    best_model_state = None
    codebert_base_path = 'data/pretrain_model/codebert-base/'
    tokenizer = RobertaTokenizer.from_pretrained(codebert_base_path)
    for epoch in range(args.num_train_epochs):
        print(f"Epoch {epoch+1}/{args.num_train_epochs}")
        # Fine-tune
        if args.tuning_strategy == "full":
            avg_loss = full_finetune(args, model, optimizer, train_dataloader, args.device, task_type)
        else:
            if task_type == "code_clone":
                module_path = "data/module_java/lr_0.001_alpha_10.0_ne_4_wrr_22.94/result/pytorch_model_try.bin"
            else:
                module_path = "data/module_python/lr_0.001_alpha_10.0_ne_4_wrr_24.15/result/pytorch_model_try.bin"
            
            avg_loss = mask_finetune(args, model, optimizer, train_dataloader, args.device, task_type, module_path)
            
        logger.info(f"Epoch {epoch+1}/{args.num_train_epochs}, Average loss: {avg_loss:.4f}")
        
        # Evaluation
        logger.info(f"Evaluating on validation set...")
        if task_type == "code_clone":
            eval_result = evaluate_clone(model, tokenizer, "Clone_detection_BigCloneBench_2/dataset/valid.txt", "./task_eval/") 
            current_metric = eval_result["eval_precision"]
        else:  # nl_code_search
            eval_result = evaluate_search(model, tokenizer, 'NL_code_search_WebQuery/CoSQA/cosqa_dev.json', "./task_eval/")
            current_metric = eval_result["acc"]
            
        logger.info(f"Validation Results: {eval_result}")
        
        # Save best model
        if current_metric > best_metric:
            best_metric = current_metric
            best_model_state = copy.deepcopy(model.state_dict())
            logger.info(f"New best model with {task_type} metric: {current_metric:.4f}")

            
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Save model
    output_path = os.path.join(
        args.output_dir, 
        f"{task_type}_stage{stage+1}_{args.tuning_strategy}.pt"
    )
    torch.save(model.state_dict(), output_path)
    logger.info(f"Model saved to {output_path}")
    
    return model

def merge_models(args, models_to_merge, stage,base_model):
    """Merge models"""
    logger.info(f"Start stage {stage+1} model merging")
    
    merging_method = MergingMethod(args.merge_method)
    
    # Ensure there is a base model to receive merged parameters
    if args.tuning_strategy == "mask":
        scaling_coefficients = [0.5, 0.5]
    else:
        scaling_coefficients = [0.5, 0.5]
    # Perform model merging
    scaling_coefficients = [args.alpha1, args.alpha2]
    merged_params = merging_method.get_merged_model(
        merged_model=base_model,
        models_to_merge=models_to_merge,
        exclude_param_names_regex=[".*classifier.*", ".*mlp.*"],
        scaling_coefficients=scaling_coefficients,
        weight_mask_rates=[0.0, 0.0],  
        models_use_deepcopy=True
    )
    
    # Save merged model parameters
    output_path = os.path.join(
        args.output_dir, 
        f"merged_stage{stage+1}_{args.merge_method}.pt"
    )
    torch.save(merged_params, output_path)
    logger.info(f"Merged model saved to {output_path}")
    
    return merged_params
def prepare_datasets(args):
    """Prepare datasets for training and validation"""
    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_model_path)
    
    # Create processed directories
    code_clone_processed_dir = os.path.join(os.path.dirname(args.code_clone_data_path.rstrip('/')), "processed")
    nl_search_processed_dir = os.path.join(os.path.dirname(args.nl_code_search_data_path.rstrip('/')), "processed")
    os.makedirs(code_clone_processed_dir, exist_ok=True)
    os.makedirs(nl_search_processed_dir, exist_ok=True)
    
    # Prepare Code Clone datasets
    logger.info("Preparing Code Clone datasets...")
    
    # Define processed file paths
    cc_train_processed_path = os.path.join(code_clone_processed_dir, "train_dataset.pt")
    cc_val_processed_path = os.path.join(code_clone_processed_dir, "val_dataset.pt")
    cc_splits_processed_path = os.path.join(code_clone_processed_dir, "train_splits.pt")
    
    # Check if processed files exist
    if os.path.exists(cc_train_processed_path) and os.path.exists(cc_val_processed_path) and os.path.exists(cc_splits_processed_path):
        logger.info("Loading cached Code Clone datasets...")
        # code_clone_train_dataset = torch.load(cc_train_processed_path)
        # code_clone_val_dataset = torch.load(cc_val_processed_path)
        code_clone_train_splits = torch.load(cc_splits_processed_path)
    else:
        # Process from scratch
        pool = multiprocessing.Pool(processes=4)
        
        # Train dataset
        code_clone_train_file = os.path.join(args.code_clone_data_path, "train.txt")
        code_clone_train_dataset = load_and_cache_examples(tokenizer, code_clone_train_file, pool=pool)
        
        # Validation dataset
        code_clone_val_file = os.path.join(args.code_clone_data_path, "valid.txt")
        code_clone_val_dataset = load_and_cache_examples(tokenizer, code_clone_val_file, pool=pool)
        
        pool.close()
        pool.join()
        
        # Split train dataset
        code_clone_train_splits = split_dataset(code_clone_train_dataset)
        
        # Save processed datasets
        torch.save(code_clone_train_dataset, cc_train_processed_path)
        torch.save(code_clone_val_dataset, cc_val_processed_path)
        torch.save(code_clone_train_splits, cc_splits_processed_path)
        
        logger.info("Code Clone datasets processed and cached.")
    
    # Prepare NL Code Search datasets
    logger.info("Preparing NL Code Search datasets...")
    
    # Define processed file paths
    nl_train_processed_path = os.path.join(nl_search_processed_dir, "train_dataset.pt")
    nl_val_processed_path = os.path.join(nl_search_processed_dir, "val_dataset.pt")
    nl_splits_processed_path = os.path.join(nl_search_processed_dir, "train_splits.pt")
    
    # Check if processed files exist
    if os.path.exists(nl_train_processed_path) and os.path.exists(nl_val_processed_path) and os.path.exists(nl_splits_processed_path):
        logger.info("Loading cached NL Code Search datasets...")
        # nl_search_train_dataset = torch.load(nl_train_processed_path)
        # nl_search_val_dataset = torch.load(nl_val_processed_path)
        nl_search_train_splits = torch.load(nl_splits_processed_path)
    else:
        # Process from scratch
        max_seq_length = tokenizer.max_len_single_sentence
        
        # Train dataset
        nl_search_train_file = os.path.join(args.nl_code_search_data_path, "cosqa_train.json")
        nl_search_train_dataset = TextDataset(tokenizer, max_seq_length, nl_search_train_file, type='train')
        
        # Validation dataset
        nl_search_val_file = os.path.join(args.nl_code_search_data_path, "cosqa_dev.json")
        nl_search_val_dataset = TextDataset(tokenizer, max_seq_length, nl_search_val_file, type='dev')
        
        # Split train dataset
        nl_search_train_splits = split_dataset(nl_search_train_dataset)
        
        # Save processed datasets
        torch.save(nl_search_train_dataset, nl_train_processed_path)
        torch.save(nl_search_val_dataset, nl_val_processed_path)
        torch.save(nl_search_train_splits, nl_splits_processed_path)
        
        logger.info("NL Code Search datasets processed and cached.")
    
    # Log dataset information
    logger.info(f"Code Clone training dataset split complete: {[len(split) for split in code_clone_train_splits]} samples")
    logger.info(f"NL Code Search training dataset split complete: {[len(split) for split in nl_search_train_splits]} samples")
    # logger.info(f"Code Clone validation dataset: {len(code_clone_val_dataset)} samples")
    # logger.info(f"NL Code Search validation dataset: {len(nl_search_val_dataset)} samples")
    
    return tokenizer, code_clone_train_splits, nl_search_train_splits
def main():
    # Parse arguments
    args = parse_args()
    set_seed(args.seed)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")
    logger.info(f"Using device: {args.device}")
    
    # Prepare datasets
    tokenizer, code_clone_train_splits, nl_search_train_splits= prepare_datasets(args)
    
    # Load pretrained model config
    config = RobertaConfig.from_pretrained(args.pretrained_model_path)
    config.num_labels = 2
    
    # Create train dataloaders
    code_clone_train_dataloaders = []
    nl_search_train_dataloaders = []

    for split_idx in range(3):
        # Code Clone train dataloader
        cc_sampler = RandomSampler(code_clone_train_splits[split_idx])
        cc_dataloader = DataLoader(
            code_clone_train_splits[split_idx], 
            sampler=cc_sampler, 
            batch_size=args.batch_size,
            num_workers=2,
            pin_memory=True
        )
        code_clone_train_dataloaders.append(cc_dataloader) 
        
        # NL Code Search train dataloader
        nl_sampler = RandomSampler(nl_search_train_splits[split_idx])
        nl_dataloader = DataLoader(
            nl_search_train_splits[split_idx],
            sampler=nl_sampler,
            batch_size=args.batch_size,
            num_workers=2,
            pin_memory=True
        )
        nl_search_train_dataloaders.append(nl_dataloader)
    # Initialize or reload base model
    base_model = RobertaModel.from_pretrained(args.pretrained_model_path)
    base_merged_model=Model_clone(base_model, config, tokenizer).to(args.device)
    base_clone_model=copy.deepcopy(base_model)
    base_search_model=copy.deepcopy(base_model)
    # Stage 1: Fine-tune using pretrained models for each task
    code_clone_model = Model_clone(base_clone_model, config, tokenizer).to(args.device)
    nl_search_model = Model_search(base_search_model, config, tokenizer).to(args.device)
    stage_iterator = tqdm(range(3), desc="Training Stages", position=0, leave=True)
    
    for stage in stage_iterator:
        logger.info(f"Starting stage {stage+1} training")
        # Fine-tune both task models
        code_clone_model = train_model(
            args, 
            code_clone_model, 
            code_clone_train_dataloaders[stage], 
            "code_clone", 
            stage
        )
        
        nl_search_model = train_model(
            args, 
            nl_search_model, 
            nl_search_train_dataloaders[stage], 
            "nl_code_search", 
            stage
        )
        
        # Merge the two task models
        models_to_merge = [code_clone_model, nl_search_model]
        merged_params = merge_models(args, models_to_merge, stage,base_merged_model)
        
        # Evaluate the merged model
        logger.info("Evaluating the merged model")
        base_merged_model.load_state_dict(merged_params, strict=False)
        # Evaluate on Code Clone task
        code_clone_model.load_state_dict(merged_params, strict=False)
        code_clone_model.to(args.device)
        cc_results = evaluate_clone(
            code_clone_model,
            tokenizer,
            os.path.join(args.code_clone_data_path, "test.txt"),
            args.output_dir
        )
        logger.info(f"Code Clone evaluation results: {cc_results}")
        
        # Evaluate on NL Code Search task
        nl_search_model.load_state_dict(merged_params, strict=False)
        nl_search_model.to(args.device)
        nl_results = evaluate_search(
            nl_search_model,
            tokenizer,
            os.path.join(args.nl_code_search_data_path, "cosqa_dev.json"),
            args.output_dir
        )
        logger.info(f"NL Code Search evaluation results: {nl_results}")
        
    logger.info("All stages of training completed!")

if __name__ == "__main__":
    main()