import os
import torch
import sys
import argparse
import random
import logging

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Subset, SequentialSampler, RandomSampler
import multiprocessing

# Add project path
sys.path.append('../../')
sys.path.append('../')
from finetune.new_optimizer import AdamWS

# Import model-related modules
from transformers import (GPTNeoForCausalLM, GPT2Tokenizer, 
                          AdamW, get_linear_schedule_with_warmup, Trainer, TrainingArguments)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from merge_methods.merging_methods import MergingMethod



# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Data preprocessing functions
def create_label_mapping(dataset):
    """Create a label mapping for mathqa dataset"""
    unique_topics = set()
    for split in dataset.keys():
        unique_topics.update(set(dataset[split]['topic']))
    return {topic: idx for idx, topic in enumerate(sorted(unique_topics))}

def preprocess_mathqa_data(examples, tokenizer, label_mapping=None, max_length=512):
    """Preprocess function for mathqa data"""
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
    
    return inputs

def preprocess_scotus_data(examples, tokenizer, max_length=512):
    """Preprocess function for scotus data"""
    inputs = tokenizer(
        examples["text"],
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors=None
    )
    
    inputs["labels"] = examples["label"]
    return inputs


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

def prepare_and_save_datasets(args):
    os.makedirs(args.output_dir, exist_ok=True)
    mathqa_dir = os.path.join(args.output_dir, "mathqa")
    scotus_dir = os.path.join(args.output_dir, "scotus")
    os.makedirs(mathqa_dir, exist_ok=True)
    os.makedirs(scotus_dir, exist_ok=True)
    
    set_seed(args.seed)
    
    tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    result = {
        "mathqa": {"num_labels": 0, "train_splits": []},
        "scotus": {"num_labels": 0, "train_splits": []}
    }
    

    logger.info("Processing MathQA dataset...")
    mathqa_dataset = load_dataset(
        'parquet',  
        data_files={
            'train': os.path.join(args.mathqa_data_path, "train-00000-of-00001.parquet"),
            'validation': os.path.join(args.mathqa_data_path, "val-00000-of-00001.parquet"),
            'test': os.path.join(args.mathqa_data_path, "test-00000-of-00001.parquet")
        }
    )
    
    label_mapping = create_label_mapping(mathqa_dataset)
    num_labels = len(label_mapping)
    result["mathqa"]["num_labels"] = num_labels
    result["mathqa"]["label_mapping"] = label_mapping
    
    processed_mathqa = mathqa_dataset.map(
        lambda x: preprocess_mathqa_data(x, tokenizer, label_mapping),
        batched=True,
        remove_columns=mathqa_dataset["train"].column_names
    )
    
    torch.save(processed_mathqa['validation'], os.path.join(mathqa_dir, "validation.pt"))
    torch.save(processed_mathqa['test'], os.path.join(mathqa_dir, "test.pt"))
    
    train_splits = split_dataset(processed_mathqa['train'], num_splits=args.num_splits)
    for i, split in enumerate(train_splits):
        torch.save(split, os.path.join(mathqa_dir, f"train_split_{i}.pt"))
        result["mathqa"]["train_splits"].append(os.path.join(mathqa_dir, f"train_split_{i}.pt"))
    
    logger.info("Processing SCOTUS dataset...")
    scotus_dataset = load_dataset(
        'parquet',
        data_files={
            'train': os.path.join(args.scotus_data_path, "train-00000-of-00001.parquet"),
            'validation': os.path.join(args.scotus_data_path, "validation-00000-of-00001.parquet"),
            'test': os.path.join(args.scotus_data_path, "test-00000-of-00001.parquet")
        }
    )
    
    num_labels = max(scotus_dataset["train"]["label"]) + 1
    result["scotus"]["num_labels"] = num_labels
    
    processed_scotus = scotus_dataset.map(
        lambda x: preprocess_scotus_data(x, tokenizer),
        batched=True,
        remove_columns=scotus_dataset["train"].column_names
    )
    

    torch.save(processed_scotus['validation'], os.path.join(scotus_dir, "validation.pt"))
    torch.save(processed_scotus['test'], os.path.join(scotus_dir, "test.pt"))
    
    train_splits = split_dataset(processed_scotus['train'], num_splits=args.num_splits)
    for i, split in enumerate(train_splits):
        torch.save(split, os.path.join(scotus_dir, f"train_split_{i}.pt"))
        result["scotus"]["train_splits"].append(os.path.join(scotus_dir, f"train_split_{i}.pt"))
    

    metadata = {
        "mathqa": {
            "num_labels": num_labels,
            "train_splits": [os.path.join(mathqa_dir, f"train_split_{i}.pt") for i in range(args.num_splits)],
            "validation": os.path.join(mathqa_dir, "validation.pt"),
            "test": os.path.join(mathqa_dir, "test.pt"),
            "label_mapping": label_mapping
        },
        "scotus": {
            "num_labels": num_labels,
            "train_splits": [os.path.join(scotus_dir, f"train_split_{i}.pt") for i in range(args.num_splits)],
            "validation": os.path.join(scotus_dir, "validation.pt"),
            "test": os.path.join(scotus_dir, "test.pt")
        }
    }
    
    torch.save(metadata, os.path.join(args.output_dir, "datasets_metadata.pt"))
    
    logger.info(f"Datasets processed and saved to {args.output_dir}")
    logger.info(f"MathQA: {len(train_splits)} train splits, validation: {len(processed_mathqa['validation'])}, test: {len(processed_mathqa['test'])}")
    logger.info(f"SCOTUS: {len(train_splits)} train splits, validation: {len(processed_scotus['validation'])}, test: {len(processed_scotus['test'])}")
    
    return metadata


def load_dataset_new(datasets_dir, stage=0, tokenizer=None, mathqa_only=False, scotus_only=False):

    import os
    import torch
    
    metadata_path = os.path.join(datasets_dir, "datasets_metadata.pt")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"no metadata: {metadata_path}")
    
    metadata = torch.load(metadata_path)
    
    num_splits = len(metadata["mathqa"]["train_splits"])
    if stage >= num_splits:
        logger.warning(f"stage {stage} over {num_splits},use stage % num_splits")
        stage = stage % num_splits
    
    result = {
        "tokenizer": tokenizer
    }
    
    if not scotus_only:
        logger.info(f"load mathqa (stage {stage})...")
        mathqa_train = torch.load(metadata["mathqa"]["train_splits"][stage])
        mathqa_val = torch.load(metadata["mathqa"]["validation"])
        mathqa_test = torch.load(metadata["mathqa"]["test"])
        
        result["mathqa"] = {
            "train": mathqa_train,
            "validation": mathqa_val,
            "test": mathqa_test,
            "num_labels": metadata["mathqa"]["num_labels"],
            "label_mapping": metadata["mathqa"].get("label_mapping", None)
        }
        
        logger.info(f"MathQA loaded - train split {stage+1}/{num_splits}, "
                   f"sample num  {len(mathqa_train)}, val: {len(mathqa_val)},test: {len(mathqa_test)}")

    if not mathqa_only:
        logger.info(f"scotus (stage {stage})...")
        scotus_train = torch.load(metadata["scotus"]["train_splits"][stage])
        scotus_val = torch.load(metadata["scotus"]["validation"])
        scotus_test = torch.load(metadata["scotus"]["test"])
        
        result["scotus"] = {
            "train": scotus_train,
            "validation": scotus_val,
            "test": scotus_test,
            "num_labels": metadata["scotus"]["num_labels"]
        }
        
        logger.info(f"SCOTUS- train split {stage+1}/{num_splits}, "
                   f"sample: {len(scotus_train)} val : {len(scotus_val)}test : {len(scotus_test)}")
    
    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./data/processed")
    parser.add_argument("--mathqa_data_path", type=str, default="TransModular_GPT/finetune/data/mathqa/")
    parser.add_argument("--scotus_data_path", type=str, default="TransModular_GPT/finetune/data/lex_glue/scotus/")
    parser.add_argument("--pretrained_model_path", type=str, default="TransModular_GPT/data/gpt-neo-125m/")
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    prepare_and_save_datasets(args)
    