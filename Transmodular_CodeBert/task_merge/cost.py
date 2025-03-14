# -*- coding: utf-8 -*-
from __future__ import absolute_import
import torch
import sys
import os
import time
import numpy as np
from tqdm import tqdm
from deepsparse import Engine
from sparseml.pytorch.utils import ModuleExporter
import torch.onnx
sys.path.append('../../')
from transformers import (RobertaModel, RobertaTokenizer, RobertaConfig)
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
from transformers import DataCollatorForLanguageModeling
from task_eval.code_clone_eval import evaluate as evaluate_clone, Model as Model_clone,load_and_cache_examples
from task_eval.nl_code_search_eval import evaluate as evaluate_search, Model as Model_search,TextDataset
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, TensorDataset
import multiprocessing
import copy
torch.multiprocessing.set_sharing_strategy('file_system')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cuda'
# def calculate_delta_sparsity(sparse_model, pretrained_model):
#     """
#     Calculate how many parameters in the difference between
#     sparse model and pretrained model are zero.
    
#     Args:
#         sparse_model: The sparse fine-tuned model
#         pretrained_model: The original pretrained model
        
#     Returns:
#         dict: Statistics about the delta sparsity
#     """
#     sparse_state = sparse_model.state_dict()
#     pretrained_state = pretrained_model.state_dict()
    
#     total_params = 0
#     zero_delta_params = 0
#     near_zero_delta_params = 0  # Parameters with very small changes (abs < 1e-6)
#     layer_stats = {}
    
#     print("\nCalculating delta sparsity (parameters unchanged from pretrained model):")
    
#     for name in sparse_state:
#         # Skip non-tensor parameters or parameters not in pretrained model
#         if name not in pretrained_state or not isinstance(sparse_state[name], torch.Tensor):
#             continue
            
#         # Skip classifier layers which are task-specific
#         if "classifier" in name or "pooler" in name or "mlp" in name or "out_proj" in name or "LayerNorm" in name:
#             continue
            
#         # Calculate the delta (difference) between sparse and pretrained
#         delta = sparse_state[name] - pretrained_state[name]
#         param_size = torch.numel(delta)
#         total_params += param_size
        
#         # Count exact zeros in delta
#         exact_zeros = torch.sum(delta == 0).item()
#         zero_delta_params += exact_zeros
        
#         # Count near-zeros in delta (very small changes)
#         near_zeros = torch.sum(torch.abs(delta) < 1e-6).item() - exact_zeros
#         near_zero_delta_params += near_zeros
        
#         # Calculate percentage for this layer
#         zero_percent = 100.0 * exact_zeros / param_size if param_size > 0 else 0
#         near_zero_percent = 100.0 * near_zeros / param_size if param_size > 0 else 0
        
#         # Store layer statistics
#         layer_stats[name] = {
#             "total_params": param_size,
#             "zero_deltas": exact_zeros,
#             "near_zero_deltas": near_zeros,
#             "zero_percent": zero_percent,
#             "near_zero_percent": near_zero_percent
#         }
        
#         # Print layers with significant sparsity
#         if zero_percent > 50:
#             print(f"  {name}: {zero_percent:.2f}% unchanged ({exact_zeros:,}/{param_size:,})")
#             if near_zero_percent > 1:
#                 print(f"    + {near_zero_percent:.2f}% near-zero changes ({near_zeros:,}/{param_size:,})")
    
#     # Calculate overall statistics
#     zero_percent_total = 100.0 * zero_delta_params / total_params if total_params > 0 else 0
#     near_zero_percent_total = 100.0 * near_zero_delta_params / total_params if total_params > 0 else 0
    
#     print(f"\nOverall delta sparsity:")
#     print(f"  Exact zeros: {zero_percent_total:.2f}% ({zero_delta_params:,}/{total_params:,})")
#     print(f"  Near zeros (< 1e-6): {near_zero_percent_total:.2f}% ({near_zero_delta_params:,}/{total_params:,})")
#     print(f"  Total unchanged or minimally changed: {zero_percent_total + near_zero_percent_total:.2f}%")
    
#     return {
#         "total_params": total_params,
#         "zero_delta_params": zero_delta_params,
#         "near_zero_delta_params": near_zero_delta_params,
#         "zero_percent": zero_percent_total,
#         "near_zero_percent": near_zero_percent_total,
#         "total_unchanged_percent": zero_percent_total + near_zero_percent_total,
#         "layer_stats": layer_stats
#     }

def calculate_sparsity(model_state_dict):
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

def load_sparse_model(base_model, sparse_model_path, config, task_type="code_clone"):
    """Load sparse model"""
    codebert_base_path = '/home/LAB/longwr/new_SeaM/Tran_SeaM/data/pretrain_model/codebert-base/'
    tokenizer = RobertaTokenizer.from_pretrained(codebert_base_path)
    if task_type == "code_clone":
        model = Model_clone(base_model, config , tokenizer)
    else:  # nl_code_search
        model = Model_search(base_model, config, tokenizer)
    
    # Load sparse model weights
    sparse_state_dict = torch.load(sparse_model_path, map_location=torch.device(device))
    model.load_state_dict(sparse_state_dict, strict=False)
    model=new_load_init_module(model,task_type)
    return model
def compare_models(dense_model, sparse_model):
    dense_state = dense_model.state_dict()
    sparse_state = sparse_model.state_dict()
    
    different_params = 0
    total_compared = 0
    
    for name in dense_state:
        if name in sparse_state and isinstance(dense_state[name], torch.Tensor):
            param_size = torch.numel(dense_state[name])
            diff_count = torch.sum(dense_state[name] != sparse_state[name]).item()
            different_params += diff_count
            total_compared += param_size
            
            if diff_count > 0:
                print(f"{name}: {100.0 * diff_count / param_size:.2f}% param different ({diff_count}/{param_size})")
    
    print(f"\n all: {100.0 * different_params / total_compared:.2f}% ({different_params}/{total_compared})")
def new_load_init_module(model,task_type="code_clone"):
    if task_type == "code_clone":
        module_path = "/home/LAB/longwr/new_SeaM/Tran_SeaM/data/module_java/lr_0.001_alpha_10.0_ne_4_wrr_22.94/result/pytorch_model_try.bin"
    else:
        module_path = "/home/LAB/longwr/new_SeaM/Tran_SeaM/data/module_python/lr_0.001_alpha_10.0_ne_4_wrr_24.15/result/pytorch_model_try.bin"
    
    # if task_type == "code_clone":
    #     module_path = "/home/LAB/longwr/new_SeaM/Tran_SeaM/data/module_java/lr_0.01_alpha_10.0_ne_2_wrr_7.22/result/pytorch_model_try.bin"
    # else:
    #     module_path = "/home/LAB/longwr/new_SeaM/Tran_SeaM/data/module_python/lr_0.01_alpha_10.0_ne_2_wrr_7.68/result/pytorch_model_try.bin"
    
    total_params = 0
    masked_params = 0
    mask_stats = {}


    module_state = torch.load(module_path, map_location=torch.device(device))
    model=model.to(device)
    model_state = model.state_dict()
    # print(f'module_state: {module_state.keys()}\n\n')
    # print(f'model_state: {model_state.keys()}\n\n')
    same_k, mask_k, diff_k = [], [], []
    pre_mask_sparsity = calculate_sparsity(model.state_dict())
    print(f"\nbefore masked: {pre_mask_sparsity['sparsity_percentage']:.2f}% zeros")
    
    for name in model_state:
        modified_name = name.replace('encoder.', 'roberta.', 1)
        if f'{modified_name}_mask' in module_state:
            same_k.append(name)
            mask = module_state[f'{modified_name}_mask']
            bin_mask = (mask > 0).float().to(device)
            ##
            # param_before = model_state[name]
            # total_in_layer = torch.numel(param_before)
            # total_params += total_in_layer


            model_state[name] = model_state[name] * bin_mask
            # zeros_after = torch.sum(model_state[name] == 0).item()

            ##
            # zeros_before = torch.sum(param_before == 0).item()
            # masked_in_layer = zeros_after - zeros_before
            # masked_params += masked_in_layer
            
            #
            # sparsity = 100.0 * masked_in_layer / total_in_layer if total_in_layer > 0 else 0
            # mask_stats[name] = {
            #     "total": total_in_layer,
            #     "masked": masked_in_layer,
            #     "sparsity": sparsity
            # }
        else:
            diff_k.append(name)

    print(f'same k: {same_k}\n\n')
    print(f'diff k: {diff_k}\n\n')
    # total_sparsity = 100.0 * masked_params / total_params if total_params > 0 else 0
    # print(f"\nall:")
    # print(f"total_params: {total_params:,}")
    # print(f"masked_params: {masked_params:,}")
    # print(f"total_sparsity: {total_sparsity:.2f}%")

    # print("\nlayer:")
    # for name, stats in mask_stats.items():
    #     if stats["sparsity"] > 0:
    #         print(f"{name}: {stats['sparsity']:.2f}% masked ({stats['masked']:,}/{stats['total']:,})")

    
    model.load_state_dict(model_state)
    post_mask_sparsity = calculate_sparsity(model.state_dict())
    print(f"after masked: {post_mask_sparsity['sparsity_percentage']:.2f}% zeros")
    return model

def load_dense_model(base_model, config, task_type="code_clone"):
    """Load dense model"""
    codebert_base_path = '/home/LAB/longwr/new_SeaM/Tran_SeaM/data/pretrain_model/codebert-base/'
    tokenizer = RobertaTokenizer.from_pretrained(codebert_base_path)
    if task_type == "code_clone":
        model = Model_clone(base_model, config, tokenizer)
        model.load_state_dict(torch.load("/home/LAB/longwr/new_SeaM/Tran_SeaM/Clone_detection_BigCloneBench_2/code/saved_models/model_fintune_20241101/checkpoint-best-f1/model.bin", map_location=torch.device(device)))
    else:  # nl_code_search
        model = Model_search(base_model, config, tokenizer)
        model.load_state_dict(torch.load("/home/LAB/longwr/new_SeaM/Tran_SeaM/NL_code_search_WebQuery/code/model_cosqa_20241031_epoch10/checkpoint-best-aver/pytorch_model.bin", map_location=torch.device(device)))
    return model


def benchmark_pytorch_model(model, data_loader, device, num_iterations=20, num_warmup=5):
    """Manually measure model inference time"""
    model.to(device)
    model.eval()
    
    times = []
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= num_iterations + num_warmup:
                break
                
            # Move data to device
            if isinstance(batch, dict):
                inputs = {k: v.to(device) for k, v in batch.items()}
            else:
                inputs = tuple(t.to(device) for t in batch)
            
            # Skip warmup iterations for timing
            if i >= num_warmup:
                start_time = time.time()
                _ = model(**inputs) if isinstance(inputs, dict) else model(*inputs)
                if device == "cuda":
                    torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms
            else:
                _ = model(**inputs) if isinstance(inputs, dict) else model(*inputs)
    
    return {
        "avg_ms_per_batch": np.mean(times),
        "std_ms_per_batch": np.std(times)
    }


# def benchmark_with_deepsparse(model, inputs, export_path, batch_size=16):
#     """Measure sparse model inference time using DeepSparse"""
#     model.to(device)
#     model.eval()
    
#     # Ensure directory exists
#     os.makedirs(os.path.dirname(export_path), exist_ok=True)
    
#     # Export model to ONNX format
#     exporter = ModuleExporter(model, os.path.dirname(export_path))
#     exporter.export_onnx(inputs, name=os.path.basename(export_path))
#     actual_batch_size = inputs[0].shape[0]
#     # Load and benchmark with DeepSparse engine
#     engine = Engine(export_path, batch_size=actual_batch_size)
    
#     # Prepare input data as numpy arrays
#     input_data = [t.numpy() for t in inputs]
    
#     # Run benchmark
#     result = engine.benchmark(input_data, 
#                              num_iterations=20, 
#                              num_warmup_iterations=5,
#                              show_progress=True)
    
#     return result.ms_per_batch


def benchmark_with_deepsparse(model, inputs, export_path, batch_size=16, data_loader=None):
    """Measure sparse model inference time using DeepSparse"""
    model.to(device)
    model.eval()
    
    class InferenceWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        

        def forward(self, code_inputs, nl_inputs):
            with torch.no_grad():
                bs = code_inputs.shape[0]
                inputs = torch.cat((code_inputs, nl_inputs), 0)
                outputs = self.model.encoder(inputs, attention_mask=inputs.ne(1))[1]
                code_vec = outputs[:bs]
                nl_vec = outputs[bs:]
                
                logits = self.model.mlp(torch.cat((nl_vec, code_vec, nl_vec-code_vec, nl_vec*code_vec), 1))
                logits = logits.view(-1)
                
                return logits
    
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    
    inference_model = InferenceWrapper(model)
    
    try:
        if len(inputs) >= 2:
            code_inputs = inputs[0]
            nl_inputs = inputs[1]
            
            print(f"Input shapes: code_inputs={code_inputs.shape}, nl_inputs={nl_inputs.shape}")
            
            torch.onnx.export(
                inference_model,
                (code_inputs, nl_inputs),  
                export_path,
                export_params=True,
                opset_version=11,  
                do_constant_folding=True,
                input_names=['code_inputs', 'nl_inputs'],
                output_names=['logits'],
                dynamic_axes={
                    'code_inputs': {0: 'batch_size'},
                    'nl_inputs': {0: 'batch_size'},
                    'logits': {0: 'batch_size'}
                }
            )
            
            print(f"ONNX model exported successfully to {export_path}")
            
            engine = Engine(export_path, batch_size=code_inputs.shape[0])
            
            input_data = [code_inputs.cpu().numpy(), nl_inputs.cpu().numpy()]
            print(f"Sending {len(input_data)} inputs to DeepSparse engine")
            
            result = engine.benchmark(
                input_data, 
                num_iterations=20, 
                num_warmup_iterations=5,
                show_progress=True
            )
            
            return result.ms_per_batch
        else:
            raise ValueError("Not enough inputs provided for the model")
        
    except Exception as e:
        print(f"ONNX export or DeepSparse benchmark failed: {e}")
        print("Falling back to PyTorch CPU benchmark...")
        
        if data_loader:
            cpu_perf = benchmark_pytorch_model(model, data_loader, "cpu", num_iterations=10, num_warmup=3)
            return cpu_perf["avg_ms_per_batch"]
        else:
            print("No data loader provided for fallback benchmarking")
            return None

def get_task_dataset_and_loader(task_type, tokenizer, batch_size=16):
    """Load and prepare dataset based on task type"""
    if task_type == "code_clone":
        # For code clone task
        pool = multiprocessing.Pool(processes=4)
        test_data_file = '/home/LAB/longwr/new_SeaM/Tran_SeaM/Clone_detection_BigCloneBench_2/dataset/test.txt'
        eval_dataset = load_and_cache_examples( tokenizer,test_data_file, pool=pool)
        pool.close()
        pool.join()
        eval_batch_size = 4
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) 
        data_loader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size,num_workers=2,pin_memory=True)

    else:  # nl_code_search
        # For nl code search task
        eval_data_path = os.path.join('/home/LAB/longwr/new_SeaM/Tran_SeaM/NL_code_search_WebQuery/CoSQA/cosqa_dev.json')
        max_seq_length = tokenizer.max_len_single_sentence
        eval_dataset = TextDataset(tokenizer,max_seq_length,eval_data_path, type='eval')
    
        # data_loader = DataLoader(processed_dataset, batch_size=batch_size, collate_fn=collator)
        eval_sampler = SequentialSampler(eval_dataset) 
        data_loader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=8, num_workers=2, pin_memory=True)

    return  data_loader


def main():
    # Base configuration
    batch_size = 16
    codebert_base_path = '/home/LAB/longwr/new_SeaM/Tran_SeaM/data/pretrain_model/codebert-base/'
    
    # Load tokenizer and config
    tokenizer = RobertaTokenizer.from_pretrained(codebert_base_path)
    config = RobertaConfig.from_pretrained(codebert_base_path)
    config.num_labels = 2  
    # Base model
    base_model = RobertaModel.from_pretrained(codebert_base_path)
    
    # Tasks to evaluate
    tasks = [ "nl_code_search"]
    
    for task_type in tasks:
        print(f"\n{'='*50}")
        print(f"Evaluating task: {task_type}")
        print(f"{'='*50}")
        
        # Create output directory
        output_dir = f"./output/{task_type}"
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Load dataset and data loader
        data_loader = get_task_dataset_and_loader(task_type, tokenizer, batch_size)
        
        # Get sample batch for ONNX export
        for batch in data_loader:
            sample_inputs = []
            if isinstance(batch, dict):
                sample_inputs = [batch['input_ids'].to(device), batch['attention_mask'].to(device)]
            else:
                sample_inputs = [t.to(device) for t in batch]
            break
        base_model_for_dense = copy.deepcopy(base_model)
        base_model_for_sparse = copy.deepcopy(base_model)
        # base_model_for_new = copy.deepcopy(base_model)
        # 2. Load dense and sparse models
        dense_model = load_dense_model(base_model_for_dense, config, task_type)
        
        # Path to sparse model weights - replace with actual path
        # if task_type == "code_clone":
        #     sparse_model_path = "/home/LAB/longwr/new_SeaM/Tran_SeaM/Clone_detection_BigCloneBench_2/code/saved_models/module_fintune_20241101/checkpoint-best-f1/model.bin"
        # else:  # nl_code_search
        #     sparse_model_path = "/home/LAB/longwr/new_SeaM/Tran_SeaM/NL_code_search_WebQuery/code/module_cosqa_20241031_epoch10/checkpoint-best-aver/pytorch_model.bin"
        
        # if task_type == "code_clone":
        #     sparse_model_path = "/home/LAB/longwr/new_SeaM/Tran_SeaM/Clone_detection_BigCloneBench_2/code/saved_models/module_fintune_wrr_7.22_20250228/checkpoint-best-f1/model.bin"
        # else:  # nl_code_search
        #     sparse_model_path = "/home/LAB/longwr/new_SeaM/Tran_SeaM/NL_code_search_WebQuery/code/module_cosqa_20250228/checkpoint-best-aver/pytorch_model.bin"
        
        if task_type == "code_clone":
            sparse_model_path = "/home/LAB/longwr/new_SeaM/Tran_SeaM/Clone_detection_BigCloneBench_2/code/saved_models/module_fintune_wrr_22.94_20250228/checkpoint-best-f1/model.bin"
        else:  # nl_code_search
            sparse_model_path = "/home/LAB/longwr/new_SeaM/Tran_SeaM/NL_code_search_WebQuery/code/module_cosqa_20250302/checkpoint-best-aver/pytorch_model.bin"

        sparse_model = load_sparse_model(base_model_for_sparse, sparse_model_path, config, task_type)
        # print("\ncompare:")
        # dense_model.to('cuda')
        # sparse_model.to('cuda')
        # print("\nAnalyzing sparse model changes from pretrained:")
        # pretrained_model = copy.deepcopy(base_model_for_new)
        # pretrained_model = Model_clone(base_model, config , tokenizer).to(device)
        # delta_stats = calculate_delta_sparsity(sparse_model, pretrained_model)
        
        # compare_models(dense_model, sparse_model)

        print("\nCalculating model sparsity:")
        dense_sparsity = calculate_sparsity(dense_model.state_dict())
        sparse_sparsity = calculate_sparsity(sparse_model.state_dict())
        
        print(f"Dense model: {dense_sparsity['sparsity_percentage']:.2f}% zeros ({dense_sparsity['zero_params']:,}/{dense_sparsity['total_params']:,})")
        print(f"Sparse model: {sparse_sparsity['sparsity_percentage']:.2f}% zeros ({sparse_sparsity['zero_params']:,}/{sparse_sparsity['total_params']:,})")
        # 3. Benchmark on CPU
        print("\nBenchmarking on CPU:")
        dense_cpu_perf = benchmark_pytorch_model(dense_model, data_loader, "cpu")
        sparse_cpu_perf = benchmark_pytorch_model(sparse_model, data_loader, "cpu")
        
        print(f"Dense model (CPU): {dense_cpu_perf['avg_ms_per_batch']:.2f} +/- {dense_cpu_perf['std_ms_per_batch']:.2f} ms/batch")
        print(f"Sparse model (CPU): {sparse_cpu_perf['avg_ms_per_batch']:.2f} +/- {sparse_cpu_perf['std_ms_per_batch']:.2f} ms/batch")
        print(f"Speedup: {dense_cpu_perf['avg_ms_per_batch']/sparse_cpu_perf['avg_ms_per_batch']:.2f}x")
        # 4. Benchmark with DeepSparse
        print("\nBenchmarking with DeepSparse:")
        export_path = f'./tmp/{task_type}/sparse_model.onnx'
        deepsparse_ms_per_batch = benchmark_with_deepsparse(sparse_model, sample_inputs, export_path, batch_size)
        
        print(f"DeepSparse acceleration: {deepsparse_ms_per_batch:.2f} ms/batch")
        print(f"Speedup vs CPU dense: {dense_cpu_perf['avg_ms_per_batch']/deepsparse_ms_per_batch:.2f}x")
        print(f"Speedup vs CPU sparse: {sparse_cpu_perf['avg_ms_per_batch']/deepsparse_ms_per_batch:.2f}x")
        
        # 5. If GPU is available, benchmark on GPU
        if torch.cuda.is_available():
            print("\nBenchmarking on GPU:")
            dense_gpu_perf = benchmark_pytorch_model(dense_model, data_loader, "cuda")
            sparse_gpu_perf = benchmark_pytorch_model(sparse_model, data_loader, "cuda")
            
            print(f"Dense model (GPU): {dense_gpu_perf['avg_ms_per_batch']:.2f} +/- {dense_gpu_perf['std_ms_per_batch']:.2f} ms/batch")
            print(f"Sparse model (GPU): {sparse_gpu_perf['avg_ms_per_batch']:.2f} +/- {sparse_gpu_perf['std_ms_per_batch']:.2f} ms/batch")
            print(f"GPU speedup: {dense_gpu_perf['avg_ms_per_batch']/sparse_gpu_perf['avg_ms_per_batch']:.2f}x")
            
            # Compare DeepSparse with GPU
            print(f"DeepSparse vs dense GPU: {dense_gpu_perf['avg_ms_per_batch']/deepsparse_ms_per_batch:.2f}x")
            print(f"DeepSparse vs sparse GPU: {sparse_gpu_perf['avg_ms_per_batch']/deepsparse_ms_per_batch:.2f}x")
        
        # 6. Evaluate model accuracy
        # print("\nEvaluating model accuracy:")
        # if task_type == "code_clone":
        #     dense_metrics = evaluate_clone(dense_model,tokenizer,"/home/LAB/longwr/new_SeaM/Tran_SeaM/Clone_detection_BigCloneBench_2/dataset/test.txt","./task_eval/")
        #     sparse_metrics = evaluate_clone(sparse_model,tokenizer,"/home/LAB/longwr/new_SeaM/Tran_SeaM/Clone_detection_BigCloneBench_2/dataset/test.txt","./task_eval/")
        # else:  # nl_code_search
        #     dense_metrics = evaluate_search(dense_model,tokenizer,'/home/LAB/longwr/new_SeaM/Tran_SeaM/NL_code_search_WebQuery/CoSQA/cosqa_dev.json',"./task_eval/")
        #     sparse_metrics = evaluate_search(sparse_model,tokenizer,'/home/LAB/longwr/new_SeaM/Tran_SeaM/NL_code_search_WebQuery/CoSQA/cosqa_dev.json',"./task_eval/")
        
        # print(f"Dense model metrics: {dense_metrics}")
        # print(f"Sparse model metrics: {sparse_metrics}")
        
        # 7. Save results
        # result_file = f"{output_dir}/benchmark_results.txt"
        # with open(result_file, "w") as f:
        #     f.write(f"Task: {task_type}\n")
        #     f.write(f"Dense model (CPU): {dense_cpu_perf['avg_ms_per_batch']:.2f}+/-{dense_cpu_perf['std_ms_per_batch']:.2f} ms/batch\n")
        #     f.write(f"Sparse model (CPU): {sparse_cpu_perf['avg_ms_per_batch']:.2f}+/-{sparse_cpu_perf['std_ms_per_batch']:.2f} ms/batch\n")
        #     f.write(f"DeepSparse: {deepsparse_ms_per_batch:.2f} ms/batch\n")
        #     if torch.cuda.is_available():
        #         f.write(f"Dense model (GPU): {dense_gpu_perf['avg_ms_per_batch']:.2f}+/-{dense_gpu_perf['std_ms_per_batch']:.2f} ms/batch\n")
        #         f.write(f"Sparse model (GPU): {sparse_gpu_perf['avg_ms_per_batch']:.2f}+/-{sparse_gpu_perf['std_ms_per_batch']:.2f} ms/batch\n")
        #     f.write(f"Dense model metrics: {dense_metrics}\n")
        #     f.write(f"Sparse model metrics: {sparse_metrics}\n")


if __name__ == "__main__":
    main()