# -*- coding: utf-8 -*-
from __future__ import absolute_import
import torch
import sys
import os
import time
import numpy as np
from tqdm import tqdm
import argparse
import logging
import copy
from transformers import GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from deepsparse import Engine  
from sparseml.pytorch.utils import ModuleExporter  
import torch.onnx  
# Import evaluation functionality and model definition
from task_eval.law_eval import evaluate_law_model, GPTNeoWithClassificationHead
from task_eval.math_eval import evaluate_math_model

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def calculate_sparsity(model_state_dict):
    """Calculate percentage of zero parameters in model"""
    total_params = 0
    zero_params = 0
    
    for param_name, param in model_state_dict.items():
        if not isinstance(param, torch.Tensor):
            continue
        param_size = torch.numel(param)
        total_params += param_size
        zero_params += torch.sum(param == 0).item()
    
    sparsity = 100.0 * zero_params / total_params if total_params > 0 else 0
    return {
        "sparsity_percentage": sparsity,
        "total_params": total_params,
        "zero_params": zero_params
    }

def load_model(model_path, model_base_path, num_classes):
    """Load model from checkpoint"""
    model = GPTNeoWithClassificationHead(model_base_path, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def apply_mask_to_model(model, mask_path):
    """Apply mask to model parameters"""
    module_state = torch.load(mask_path, map_location=device)
    model_state = model.state_dict()
    
    pre_mask_sparsity = calculate_sparsity(model_state)
    logger.info(f"Before masking: {pre_mask_sparsity['sparsity_percentage']:.2f}% zeros")
    
    for name in model_state:
        modified_name = name.replace("base_model.", "")
        if f'{modified_name}_mask' in module_state:
            mask = module_state[f'{modified_name}_mask']
            bin_mask = (mask > 0).float().to(device)
            model_state[name] = model_state[name] * bin_mask
    
    model.load_state_dict(model_state)
    post_mask_sparsity = calculate_sparsity(model_state)
    logger.info(f"After masking: {post_mask_sparsity['sparsity_percentage']:.2f}% zeros")
    
    return model

def get_task_dataset(task_type, tokenizer, batch_size=8):
    """Load dataset for specified task"""
    if task_type == "math":
        dataset_path = "TransModular_GPT/fintune/data/mathqa/"
        dataset = load_dataset(
            'parquet',
            data_files={'test': os.path.join(dataset_path, "test-00000-of-00001.parquet")}
        )
        
        # Create label mapping for math task
        unique_topics = set(dataset['test']['topic'])
        label_mapping = {topic: idx for idx, topic in enumerate(sorted(unique_topics))}
        
        # Preprocess data
        def preprocess(examples):
            inputs = tokenizer(
                examples['question'],
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors=None
            )
            inputs['labels'] = [label_mapping[topic] for topic in examples['topic']]
            return inputs
        
        tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=dataset["test"].column_names)
    
    elif task_type == "law":
        dataset_path = "TransModular_GPT/fintune/data/lex_glue/scotus/"
        dataset = load_dataset(
            'parquet',
            data_files={'test': os.path.join(dataset_path, "test-00000-of-00001.parquet")}
        )
        
        # Preprocess data
        def preprocess(examples):
            inputs = tokenizer(
                examples['text'],
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors=None
            )
            inputs['labels'] = examples['label']
            return inputs
        
        tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=dataset["test"].column_names)
    
    # Create dataloader

    dataloader = DataLoader(
        tokenized_dataset['test'], 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=tokenized_dataset['test'].collate(return_tensors=True)
    )
    
    logger.info(f"Loaded {task_type} dataset: {len(tokenized_dataset['test'])} test samples")
    return dataloader

def benchmark_pytorch_model(model, data_loader, device_name, num_iterations=20, num_warmup=5):
    """Measure model inference time"""
    model.to(device_name)
    model.eval()
    
    batch_times = []
    batch_sizes = []
    tokens_processed = []
    
    with torch.no_grad():
        # Use fewer iterations if dataset is small
        max_batches = len(data_loader)
        actual_iterations = min(num_iterations, max_batches - num_warmup)
        
        if actual_iterations <= 0:
            logger.warning(f"Not enough batches for benchmarking! Available: {max_batches}, need: {num_warmup + 1}")
            return None
        
        logger.info(f"Running benchmark: {num_warmup} warmup + {actual_iterations} timed iterations")
        
        for i, batch in enumerate(data_loader):
            if i >= num_warmup + actual_iterations:
                break
            
            if isinstance(batch, dict):
                inputs = {k: v.to(device_name) for k, v in batch.items()}
                batch_size = inputs['input_ids'].size(0) if 'input_ids' in inputs else 1
                tokens_count = inputs['input_ids'].size(0) * inputs['input_ids'].size(1) if 'input_ids' in inputs else 0
            elif isinstance(batch, list) or isinstance(batch, tuple):
                inputs = [t.to(device_name) for t in batch]
                batch_size = inputs[0].size(0) if len(inputs) > 0 else 1
                tokens_count = inputs[0].size(0) * inputs[0].size(1) if len(inputs) > 0 else 0
            else:
                logger.warning(f"Unexpected batch type: {type(batch)}")
                continue
            
            # Warmup iterations
            if i < num_warmup:
                _ = model(**inputs) if isinstance(inputs, dict) else model(*inputs)
                continue
            
            # Timed iterations
            torch.cuda.synchronize() if device_name == "cuda" else None
            start_time = time.time()
            _ = model(**inputs) if isinstance(inputs, dict) else model(*inputs)
            torch.cuda.synchronize() if device_name == "cuda" else None
            end_time = time.time()
            
            # Record measurements
            batch_time_ms = (end_time - start_time) * 1000  # Convert to ms
            batch_times.append(batch_time_ms)
            batch_sizes.append(batch_size)
            tokens_processed.append(tokens_count)
            
            if (i - num_warmup + 1) % 5 == 0:
                logger.info(f"  Batch {i-num_warmup+1}/{actual_iterations}: {batch_time_ms:.2f} ms")
    
    if not batch_times:
        return None
    
    # Calculate statistics
    avg_ms_per_batch = np.mean(batch_times)
    std_ms_per_batch = np.std(batch_times)
    avg_tokens_per_second = np.sum(tokens_processed) / (np.sum(batch_times) / 1000) if np.sum(batch_times) > 0 else 0
    
    return {
        "avg_ms_per_batch": avg_ms_per_batch,
        "std_ms_per_batch": std_ms_per_batch, 
        "avg_tokens_per_second": avg_tokens_per_second,
        "batch_size": np.mean(batch_sizes)
    }

def evaluate_accuracy(model, data_loader, device_name):
    """Evaluate model accuracy"""
    model.to(device_name)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            inputs = {k: v.to(device_name) for k, v in batch.items()}
            labels = inputs.pop('labels')
            
            outputs = model(**inputs)
            logits = outputs['logits']
            
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    acc = accuracy_score(all_labels, all_preds)
    return {"accuracy": acc}
def benchmark_with_deepsparse(model, inputs, export_path, batch_size=8, data_loader=None):
    """Measure sparse model inference time using DeepSparse"""
    model.to(device)
    model.eval()
    
    class GPTNeoInferenceWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.hidden_size = model.hidden_size
        
        def forward(self, input_ids, attention_mask):
            with torch.no_grad():
                outputs = self.model.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                last_hidden_state = outputs.hidden_states[-1]
                sentence_representation = torch.mean(last_hidden_state, dim=1)
                
                layer_norm = torch.nn.LayerNorm(self.hidden_size).to(sentence_representation.device)
                sentence_representation = layer_norm(sentence_representation)
                
                logits = self.model.classification_head(sentence_representation)
                
                return logits
    
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    
    inference_model = GPTNeoInferenceWrapper(model)
    
    try:
        if isinstance(inputs, dict) and 'input_ids' in inputs and 'attention_mask' in inputs:
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            
            logger.info(f"Input shapes: input_ids={input_ids.shape}, attention_mask={attention_mask.shape}")
            
            torch.onnx.export(
                inference_model,
                (input_ids, attention_mask),
                export_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input_ids', 'attention_mask'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size'},
                    'attention_mask': {0: 'batch_size'},
                    'logits': {0: 'batch_size'}
                }
            )
            
            logger.info(f"ONNX model exported to: {export_path}")
            
            engine = Engine(export_path, batch_size=input_ids.shape[0])
            
            input_data = [input_ids.cpu().numpy(), attention_mask.cpu().numpy()]
            logger.info(f"Sending {len(input_data)} inputs to DeepSparse engine")
            
            result = engine.benchmark(
                input_data,
                num_iterations=20,
                num_warmup_iterations=5,
                show_progress=True
            )
            
            return result.ms_per_batch
        elif isinstance(inputs, (list, tuple)) and len(inputs) >= 2:
            raise NotImplementedError("List-based inputs for GPT-Neo not yet implemented for DeepSparse")
            
        else:
            raise ValueError("Unexpected input format for DeepSparse benchmarking")
            
    except Exception as e:
        logger.error(f"DeepSparse benchmark failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        logger.info("Falling back to PyTorch CPU benchmark...")
        
        if data_loader:
            cpu_perf = benchmark_pytorch_model(model, data_loader, "cpu", num_iterations=10, num_warmup=3)
            return cpu_perf["avg_ms_per_batch"]
        else:
            logger.warning("No data loader provided for fallback benchmarking")
            return None
def main():
    parser = argparse.ArgumentParser(description="Benchmark GPT-Neo model inference")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--iterations", type=int, default=20, help="Number of timing iterations")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--tasks", nargs='+', default=["math", "law"], 
                        help="Tasks to benchmark (math, law, or both)")
    parser.add_argument("--cpu_only", action="store_true", help="Only run CPU benchmarks")
    args = parser.parse_args()
    
    # Base paths
    base_model_path = 'TransModular_GPT/data/gpt-neo-125m/'
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Define model paths
    model_paths = {
        "math": {
            "dense": "TransModular_GPT/fintune/save_model/mathqa/lr5e-05_bs4_e2/best_model/pytorch_model.bin",
            "sparse": "TransModular_GPT/fintune/save_model_with_mask_0.25/mathqa/lr5e-05_bs4_e2/best_model/pytorch_model.bin"
        },
        "law": {
            "dense": "TransModular_GPT/fintune/save_model/law/scotus/lr5e-05_bs4_e4/best_model/pytorch_model.bin", 
            "sparse": "TransModular_GPT/fintune/save_model_with_mask_0.25/law/scotus/lr5e-05_bs4_e4/best_model/pytorch_model.bin"
        }
    }
    
    mask_paths = {
        "math": "TransModular_GPT/data/module_math/lr_0.005_alpha_10.0_bs_4_time_20250228_022217/model_wrr_0.25/pytorch_model.bin",
        "law": "TransModular_GPT/data/module_law/lr_0.005_alpha_10.0_bs_4_time_20250227_104430/model_wrr_0.25/pytorch_model.bin"
    }
    
    num_classes = {
        "math": 25,
        "law": 13
    }
    
    # Create output directory
    output_dir = "./output/inference_benchmark"
    os.makedirs(output_dir, exist_ok=True)
    
    # Benchmark results
    results = {}
    
    # Run benchmarks for each task
    for task_type in args.tasks:
        logger.info(f"\n{'='*50}")
        logger.info(f"Benchmarking task: {task_type}")
        logger.info(f"{'='*50}")
        
        # Task-specific output directory
        task_output_dir = f"{output_dir}/{task_type}"
        os.makedirs(task_output_dir, exist_ok=True)
        
        # Load dataset and create dataloader
        dataloader = get_task_dataset(task_type, tokenizer, args.batch_size)
        
        # Load dense model
        logger.info(f"Loading dense model for {task_type} task...")
        dense_model = load_model(
            model_paths[task_type]["dense"], 
            base_model_path, 
            num_classes[task_type]
        )
        
        # Load sparse model
        logger.info(f"Loading sparse model for {task_type} task...")
        sparse_model = load_model(
            model_paths[task_type]["sparse"], 
            base_model_path, 
            num_classes[task_type]
        )
        
        # Apply additional masking to ensure sparsity
        logger.info(f"Applying mask to sparse model...")
        sparse_model = apply_mask_to_model(sparse_model, mask_paths[task_type])
        
        # Calculate model sparsity
        dense_sparsity = calculate_sparsity(dense_model.state_dict())
        sparse_sparsity = calculate_sparsity(sparse_model.state_dict())
        
        logger.info(f"Dense model: {dense_sparsity['sparsity_percentage']:.2f}% zeros")
        logger.info(f"Sparse model: {sparse_sparsity['sparsity_percentage']:.2f}% zeros")
        
        # Benchmark on CPU
        logger.info("\nBenchmarking on CPU...")
        dense_cpu_perf = benchmark_pytorch_model(
            dense_model, dataloader, "cpu", 
            num_iterations=args.iterations, 
            num_warmup=args.warmup
        )
        sparse_cpu_perf = benchmark_pytorch_model(
            sparse_model, dataloader, "cpu", 
            num_iterations=args.iterations, 
            num_warmup=args.warmup
        )
        
        if dense_cpu_perf and sparse_cpu_perf:
            cpu_speedup = dense_cpu_perf['avg_ms_per_batch'] / sparse_cpu_perf['avg_ms_per_batch']
            logger.info(f"Dense model (CPU): {dense_cpu_perf['avg_ms_per_batch']:.2f} {dense_cpu_perf['std_ms_per_batch']:.2f} ms/batch")
            logger.info(f"Sparse model (CPU): {sparse_cpu_perf['avg_ms_per_batch']:.2f} {sparse_cpu_perf['std_ms_per_batch']:.2f} ms/batch")
            logger.info(f"CPU speedup: {cpu_speedup:.2f}x")
            logger.info(f"Dense tokens/sec: {dense_cpu_perf['avg_tokens_per_second']:.2f}")
            logger.info(f"Sparse tokens/sec: {sparse_cpu_perf['avg_tokens_per_second']:.2f}")
        
        
        deepsparse_perf = None

        logger.info("\nBenchmarking with DeepSparse...")
        
        sample_inputs = {}
        for batch in dataloader:
            sample_inputs = batch
            break
        
        export_path = f'./tmp/{task_type}/sparse_model.onnx'
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        
        deepsparse_ms_per_batch = benchmark_with_deepsparse(
            sparse_model, sample_inputs, export_path, 
            batch_size=args.batch_size, data_loader=dataloader
        )
        
        if deepsparse_ms_per_batch:
            deepsparse_perf = {"avg_ms_per_batch": deepsparse_ms_per_batch}
            
            logger.info(f"DeepSparse: {deepsparse_ms_per_batch:.2f} ms/batch")
            logger.info(f"DeepSparse vs CPU dense: {dense_cpu_perf['avg_ms_per_batch']/deepsparse_ms_per_batch:.2f}x")
            logger.info(f"DeepSparse vs CPU sparse: {sparse_cpu_perf['avg_ms_per_batch']/deepsparse_ms_per_batch:.2f}x")
            
 
        # Evaluate model accuracy
        logger.info("\nEvaluating model accuracy...")
        dense_metrics = evaluate_accuracy(dense_model, dataloader, "cuda" if torch.cuda.is_available() else "cpu")
        sparse_metrics = evaluate_accuracy(sparse_model, dataloader, "cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Dense model accuracy: {dense_metrics['accuracy']:.4f}")
        logger.info(f"Sparse model accuracy: {sparse_metrics['accuracy']:.4f}")
        logger.info(f"Accuracy difference: {sparse_metrics['accuracy'] - dense_metrics['accuracy']:.4f}")
        

    
    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info(f"BENCHMARK SUMMARY")
    logger.info(f"{'='*50}")
    
    for task_type, result in results.items():
        logger.info(f"\n{task_type.upper()}:")
        logger.info(f"  Sparsity: {result['sparse_sparsity']:.2f}%")
        logger.info(f"  CPU Speedup: {result['cpu_performance']['speedup']:.2f}x")
        
        if not args.cpu_only and torch.cuda.is_available() and "gpu_performance" in result:
            logger.info(f"  GPU Speedup: {result['gpu_performance']['speedup']:.2f}x")


if __name__ == "__main__":
    main()