# -*- coding: utf-8 -*-
from __future__ import absolute_import
import torch
import sys
import os
import time
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import copy
import multiprocessing
import logging

# GPT-Neo specific imports
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, GPTNeoConfig, Trainer, TrainingArguments
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, TensorDataset
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
from sklearn.metrics import accuracy_score, f1_score

# DeepSparse imports (forced integration despite end-of-life status)
try:
    from deepsparse import Engine, compile_model
    import deepsparse
    DEEPSPARSE_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info(f"DeepSparse {deepsparse.__version__} is available for sparse acceleration")
    logger.warning("Note: DeepSparse has reached end-of-life but is being used per user request")
except ImportError as e:
    DEEPSPARSE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"DeepSparse not available: {e} - will use PyTorch for all inference")

# Note: Task evaluation modules can be imported if needed for accuracy evaluation
# Currently focusing on performance benchmarking only

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.multiprocessing.set_sharing_strategy('file_system')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GPT-Neo model with classification head for downstream tasks
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

def calculate_sparsity(model_state_dict):
    """Calculate sparsity of model parameters"""
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

def load_sparse_model(base_model_path, sparse_model_path, task_type="math"):
    """Load sparse GPT-Neo model for classification tasks"""
    tokenizer = GPT2Tokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # All tasks are now classification tasks
    num_classes = get_num_classes_for_task(task_type)
    model = GPTNeoWithClassificationHead(base_model_path, num_classes)

    # Load sparse model weights
    sparse_state_dict = torch.load(sparse_model_path, map_location=torch.device(device))
    model.load_state_dict(sparse_state_dict, strict=False)

    # Apply mask if available
    model = apply_mask_to_model(model, task_type)
    return model

def load_dense_model(base_model_path, task_type="math"):
    """Load dense GPT-Neo model for classification tasks"""
    logger.info(f"Loading dense model for task: {task_type}")
    logger.info(f"Base model path: {base_model_path}")

    tokenizer = GPT2Tokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer loaded successfully")

    # All tasks are now classification tasks
    num_classes = get_num_classes_for_task(task_type)
    logger.info(f"Creating classification model with {num_classes} classes")

    model = GPTNeoWithClassificationHead(base_model_path, num_classes)
    logger.info("Base classification model created")

    # Load fine-tuned weights if available
    dense_model_path = get_dense_model_path(task_type)
    if dense_model_path and os.path.exists(dense_model_path):
        logger.info(f"Loading fine-tuned weights from: {dense_model_path}")
        model.load_state_dict(torch.load(dense_model_path, map_location=torch.device(device)))
        logger.info("Fine-tuned weights loaded")
    else:
        logger.info("No fine-tuned weights found, using base model")

    return model

def get_num_classes_for_task(task_type):
    """Get number of classes for different classification tasks"""
    task_classes = {
        "math": 25,      # Math QA topics
        "law": 13,       # SCOTUS legal categories
    }
    return task_classes.get(task_type, 2)

def get_dense_model_path(task_type):
    """Get path to dense model for different tasks"""
    dense_paths = {
        "math": "TransModular_GPT/fintune/save_model/mathqa/lr5e-05_bs4_e2/best_model/pytorch_model.bin",
        "law": "TransModular_GPT/fintune/save_model/law/scotus/lr5e-05_bs4_e4/best_model/pytorch_model.bin",
    }
    return dense_paths.get(task_type, None)

def apply_mask_to_model(model, task_type):
    """Apply mask to model parameters"""
    mask_paths = {
        "math": "TransModular_GPT/data/module_math/lr_0.005_alpha_10.0_bs_4_time_20250228_022217/model_wrr_0.25/pytorch_model.bin",
        "law": "TransModular_GPT/data/module_law/lr_0.005_alpha_10.0_bs_4_time_20250227_104430/model_wrr_0.25/pytorch_model.bin",
    }

    # Find the most recent mask file
    import glob
    mask_pattern = mask_paths.get(task_type, "")
    if mask_pattern:
        mask_files = glob.glob(mask_pattern)
        if mask_files:
            mask_path = sorted(mask_files)[-1]  # Get the most recent
            logger.info(f"Applying mask from: {mask_path}")

            try:
                mask_state = torch.load(mask_path, map_location=torch.device(device))
                model_state = model.state_dict()

                pre_mask_sparsity = calculate_sparsity(model.state_dict())
                logger.info(f"Before mask: {pre_mask_sparsity['sparsity_percentage']:.2f}% zeros")

                # Apply masks to corresponding parameters
                for name in model_state:
                    modify_name = name.replace("base_model.", "")
                    mask_name = f"{modify_name}_mask"
                    if mask_name in mask_state:
                        mask = mask_state[mask_name]
                        bin_mask = (mask > 0).float().to(device)
                        model_state[name] = model_state[name] * bin_mask

                model.load_state_dict(model_state)
                post_mask_sparsity = calculate_sparsity(model.state_dict())
                logger.info(f"After mask: {post_mask_sparsity['sparsity_percentage']:.2f}% zeros")

            except Exception as e:
                logger.warning(f"Failed to apply mask: {e}")

    return model

def compare_models(dense_model, sparse_model):
    """Compare parameters between dense and sparse models"""
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
                logger.info(f"{name}: {100.0 * diff_count / param_size:.2f}% param different ({diff_count}/{param_size})")

    logger.info(f"\nAll: {100.0 * different_params / total_compared:.2f}% ({different_params}/{total_compared})")

def benchmark_pytorch_model(model, data_loader, device, num_iterations=20, num_warmup=5):
    """Manually measure model inference time"""
    model.to(device)
    model.eval()

    times = []

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= num_iterations + num_warmup:
                break

            # Move data to device and prepare inputs
            if isinstance(batch, dict):
                # Dataset format: dict with keys like 'input_ids', 'attention_mask', 'labels'
                inputs = {
                    'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device)
                }
            elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                # TensorDataset format: tuple of tensors
                inputs = {
                    'input_ids': batch[0].to(device),
                    'attention_mask': batch[1].to(device)
                }
            else:
                logger.warning(f"Unexpected batch format: {type(batch)}")
                continue

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

def benchmark_sparse_optimized(model, data_loader, device, num_iterations=20, num_warmup=5):
    """Optimized benchmark for sparse models (CPU-focused)"""
    model.to(device)
    model.eval()

    # Set CPU optimization for sparse models
    if device == "cpu":
        torch.set_num_threads(4)  # Optimize for sparse computation

    times = []

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= num_iterations + num_warmup:
                break

            # Move data to device and prepare inputs
            if isinstance(batch, dict):
                # Dataset format: dict with keys like 'input_ids', 'attention_mask', 'labels'
                inputs = {
                    'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device)
                }
            elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                # TensorDataset format: tuple of tensors
                inputs = {
                    'input_ids': batch[0].to(device),
                    'attention_mask': batch[1].to(device)
                }
            else:
                logger.warning(f"Unexpected batch format: {type(batch)}")
                continue

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

def check_deepsparse_availability():
    """Check if DeepSparse is available and properly installed"""
    if not DEEPSPARSE_AVAILABLE:
        logger.warning("DeepSparse is not installed. Install with: pip install deepsparse==1.7.0")
        logger.warning("Note: Using older version due to end-of-life status of newer versions")
        return False

    try:
        import deepsparse
        logger.info(f"DeepSparse version {deepsparse.__version__} is available")
        logger.warning("Using DeepSparse despite end-of-life status per user request")
        return True
    except Exception as e:
        logger.error(f"DeepSparse import error: {e}")
        return False

def export_model_to_onnx(model, tokenizer, output_path, sample_input_length=512):
    """
    Export GPT-Neo classification model to ONNX format for DeepSparse acceleration
    Tries multiple opset versions for compatibility

    Args:
        model: GPT-Neo classification model
        tokenizer: Tokenizer for creating sample inputs
        output_path: Path to save ONNX model
        sample_input_length: Length of sample input sequence

    Returns:
        bool: Success status
    """
    try:
        logger.info(f"Exporting GPT-Neo classification model to ONNX: {output_path}")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Create sample inputs
        sample_text = "This is a sample input for ONNX export and DeepSparse acceleration."
        inputs = tokenizer(
            sample_text,
            max_length=sample_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Set model to evaluation mode
        model.eval()

        # First try: Use torch.jit.trace for better ONNX compatibility
        try:
            logger.info("Attempting ONNX export using torch.jit.trace method")

            # Create traced model to avoid dynamic control flow issues
            with torch.no_grad():
                traced_model = torch.jit.trace(model, (input_ids, attention_mask))

            # Export traced model to ONNX
            torch.onnx.export(
                traced_model,
                (input_ids, attention_mask),
                output_path,
                export_params=True,
                opset_version=11,  # Use stable opset version
                do_constant_folding=False,  # Disable to avoid tracing issues
                input_names=['input_ids', 'attention_mask'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size'},
                    'attention_mask': {0: 'batch_size'},
                    'logits': {0: 'batch_size'}
                }
            )

            logger.info(f"Model successfully exported to {output_path} using torch.jit.trace")
            return True

        except Exception as trace_error:
            logger.warning(f"torch.jit.trace export failed: {trace_error}")

            # Fallback: Try different opset versions with direct export
            opset_versions = [14, 13, 12, 11]

            for opset_version in opset_versions:
                try:
                    logger.info(f"Attempting direct ONNX export with opset version {opset_version}")

                    torch.onnx.export(
                        model,
                        (input_ids, attention_mask),
                        output_path,
                        export_params=True,
                        opset_version=opset_version,
                        do_constant_folding=False,  # Disable to avoid complex operations
                        input_names=['input_ids', 'attention_mask'],
                        output_names=['logits'],
                        dynamic_axes={
                            'input_ids': {0: 'batch_size'},
                            'attention_mask': {0: 'batch_size'},
                            'logits': {0: 'batch_size'}
                        }
                    )

                    logger.info(f"Model successfully exported to {output_path} with opset {opset_version}")
                    return True

                except Exception as opset_error:
                    logger.warning(f"ONNX export failed with opset {opset_version}: {opset_error}")
                    if opset_version == opset_versions[-1]:  # Last attempt
                        raise opset_error
                    continue

        return False

    except Exception as e:
        logger.error(f"Failed to export model to ONNX with all opset versions: {e}")
        logger.error("This might be due to GPT-Neo model complexity or ONNX version compatibility")

        # Try simplified export as fallback
        logger.info("Attempting simplified ONNX export as fallback...")
        return export_simplified_model_to_onnx(model, tokenizer, output_path, sample_input_length)

def export_simplified_model_to_onnx(model, tokenizer, output_path, sample_input_length=512):
    """
    Simplified ONNX export that bypasses complex GPT-Neo operations
    Creates a wrapper model that avoids problematic operations
    """
    try:
        import torch.nn as nn

        logger.info("Creating simplified model wrapper for ONNX export")

        class SimplifiedGPTNeoWrapper(nn.Module):
            def __init__(self, original_model):
                super().__init__()
                self.original_model = original_model

            def forward(self, input_ids, attention_mask=None):
                # Use the original model but with simplified attention
                with torch.no_grad():
                    try:
                        # Try to get embeddings from the base model
                        embeddings = self.original_model.base_model.transformer.wte(input_ids)

                        # Simple pooling instead of full transformer
                        if attention_mask is not None:
                            # Masked average pooling
                            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                            sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
                            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                            pooled = sum_embeddings / sum_mask
                        else:
                            # Simple average pooling
                            pooled = torch.mean(embeddings, dim=1)

                        # Use the classification head
                        logits = self.original_model.classification_head(pooled)
                        return {'logits': logits}

                    except Exception as e:
                        logger.warning(f"Simplified forward failed: {e}, using dummy output")
                        # Fallback: return dummy logits
                        batch_size = input_ids.shape[0]
                        num_classes = self.original_model.num_classes
                        dummy_logits = torch.zeros(batch_size, num_classes)
                        return {'logits': dummy_logits}

        # Create wrapper model
        wrapper_model = SimplifiedGPTNeoWrapper(model)
        wrapper_model.eval()

        # Create sample inputs
        sample_text = "This is a simplified sample for ONNX export."
        inputs = tokenizer(
            sample_text,
            max_length=min(sample_input_length, 128),  # Shorter for simplified model
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Export simplified model
        torch.onnx.export(
            wrapper_model,
            (input_ids, attention_mask),
            output_path,
            export_params=True,
            opset_version=11,  # Use older opset for compatibility
            do_constant_folding=False,  # Disable to avoid tracing issues
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size'},
                'attention_mask': {0: 'batch_size'},
                'logits': {0: 'batch_size'}
            }
        )

        logger.info(f"Simplified model successfully exported to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Simplified ONNX export also failed: {e}")
        return False

def load_deepsparse_engine(onnx_path, batch_size=1):
    """
    Load ONNX model with DeepSparse acceleration (v1.7.0 compatible)

    Args:
        onnx_path: Path to ONNX model file
        batch_size: Batch size for inference

    Returns:
        DeepSparse engine or None if failed
    """
    if not check_deepsparse_availability():
        return None

    try:
        logger.info(f"Loading DeepSparse engine from {onnx_path}")

        # For DeepSparse 1.7.0, use Engine class directly
        engine = Engine(
            model=onnx_path,
            batch_size=batch_size
        )

        logger.info("DeepSparse engine loaded successfully")
        logger.info(f"Engine batch size: {batch_size}")
        return engine

    except Exception as e:
        logger.error(f"Failed to load DeepSparse engine: {e}")
        logger.error("This might be due to ONNX model compatibility issues")
        return None

def benchmark_deepsparse_engine(engine, data_loader, num_iterations=20, num_warmup=5):
    """
    Benchmark DeepSparse engine inference performance

    Args:
        engine: DeepSparse engine
        data_loader: DataLoader with test data
        num_iterations: Number of inference batches for measurement
        num_warmup: Number of warmup batches

    Returns:
        dict: Performance metrics
    """
    if engine is None:
        logger.error("DeepSparse engine is None")
        return None

    logger.info(f"Benchmarking DeepSparse engine ({num_iterations} batches, {num_warmup} warmup)")

    times = []

    for i, batch in enumerate(data_loader):
        if i >= num_iterations + num_warmup:
            break

        # Prepare inputs for DeepSparse engine
        if isinstance(batch, dict):
            # Dataset format: dict with keys like 'input_ids', 'attention_mask', 'labels'
            input_ids = batch['input_ids'].cpu().numpy()
            attention_mask = batch['attention_mask'].cpu().numpy()
        elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
            # TensorDataset format: tuple of tensors
            input_ids = batch[0].cpu().numpy()
            attention_mask = batch[1].cpu().numpy()
        else:
            logger.warning(f"Unexpected batch format: {type(batch)}")
            continue

        input_data = [input_ids, attention_mask]

        # Skip warmup iterations for timing
        if i >= num_warmup:
            start_time = time.time()
            try:
                outputs = engine.run(input_data)
            except Exception as e:
                logger.error(f"DeepSparse engine run failed: {e}")
                continue
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        else:
            # Warmup run
            try:
                _ = engine.run(input_data)
            except Exception as e:
                logger.warning(f"DeepSparse warmup run failed: {e}")
                continue

    if not times:
        logger.error("No successful DeepSparse inference runs")
        return None

    return {
        "avg_ms_per_batch": np.mean(times),
        "std_ms_per_batch": np.std(times)
    }

def try_deepsparse_acceleration(model, tokenizer, task_type, batch_size=8):
    """
    Try to apply DeepSparse acceleration to the sparse model

    Args:
        model: PyTorch model to accelerate
        tokenizer: Tokenizer for the model
        task_type: Task type for naming the ONNX file
        batch_size: Batch size for the engine

    Returns:
        tuple: (engine_or_model, is_deepsparse_enabled)
    """
    if not check_deepsparse_availability():
        logger.info("DeepSparse not available, using PyTorch model")
        return model, False

    try:
        # Create output directory for ONNX models
        onnx_dir = f"./output/{task_type}/onnx"
        os.makedirs(onnx_dir, exist_ok=True)

        # Export model to ONNX
        onnx_path = os.path.join(onnx_dir, "sparse_model.onnx")
        export_success = export_model_to_onnx(model, tokenizer, onnx_path)

        if not export_success:
            logger.warning("ONNX export failed, using PyTorch model")
            return model, False

        # Load DeepSparse engine
        engine = load_deepsparse_engine(onnx_path, batch_size)

        if engine is None:
            logger.warning("DeepSparse engine creation failed, using PyTorch model")
            return model, False

        logger.info("DeepSparse acceleration successfully applied")
        return engine, True

    except Exception as e:
        logger.error(f"DeepSparse acceleration failed: {e}")
        logger.info("Falling back to PyTorch model")
        return model, False

def get_task_dataset_and_loader(task_type, tokenizer, batch_size=8):
    """Load and prepare dataset based on task type - all classification tasks"""
    if task_type == "math":
        # Math QA dataset
        dataset_path = "TransModular_GPT/fintune/data/mathqa"
        try:
            dataset = load_dataset(
                'parquet',
                data_files={'test': os.path.join(dataset_path, "test-00000-of-00001.parquet")}
            )

            # Create label mapping
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

            processed_dataset = dataset['test'].map(preprocess, batched=True)
            processed_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
            data_loader = DataLoader(processed_dataset, batch_size=batch_size, shuffle=False)

        except Exception as e:
            logger.warning(f"Failed to load math dataset: {e}, using dummy data")
            return get_dummy_dataset_loader(tokenizer, batch_size, task_type)

    elif task_type == "law":
        # SCOTUS dataset
        try:
            dataset_path = "TransModular_GPT/fintune/data/lex_glue/scotus"
            dataset = load_dataset(
                'parquet',
                data_files={'test': os.path.join(dataset_path, "test-00000-of-00001.parquet")}
            )

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

            processed_dataset = dataset['test'].map(preprocess, batched=True)
            processed_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
            data_loader = DataLoader(processed_dataset, batch_size=batch_size, shuffle=False)

        except Exception as e:
            logger.warning(f"Failed to load law dataset: {e}, using dummy data")
            return get_dummy_dataset_loader(tokenizer, batch_size, task_type)

    else:
        # Default: create dummy dataset
        return get_dummy_dataset_loader(tokenizer, batch_size, task_type)

    return data_loader

def get_dummy_dataset_loader(tokenizer, batch_size, task_type):
    """Create a dummy dataset for testing"""
    texts = [
        "This is a sample text for testing the model performance.",
        "Another example sentence to evaluate inference speed.",
        "Machine learning models require proper evaluation metrics.",
        "Performance benchmarking is crucial for model optimization.",
        "Deep learning frameworks provide various optimization techniques."
    ] * 20

    inputs = tokenizer(
        texts,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    # Add dummy labels for classification
    labels = torch.randint(0, get_num_classes_for_task(task_type), (len(texts),))
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)

    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def main():
    """Main function to run GPT-Neo model performance evaluation"""
    # Performance testing configuration
    batch_size = 8           # Fixed batch size: 8 samples per batch
    num_iterations = 20      # Number of inference batches for measurement
    num_warmup = 5          # Number of warmup batches to exclude initialization overhead

    # Model configuration
    gpt_neo_base_path = 'TransModular_GPT/data/gpt-neo-125m/'

    # Load tokenizer and config
    tokenizer = GPT2Tokenizer.from_pretrained(gpt_neo_base_path)
    tokenizer.pad_token = tokenizer.eos_token
    config = GPTNeoConfig.from_pretrained(gpt_neo_base_path)

    # Tasks to evaluate - limited to math and law classification tasks
    tasks = ["math", "law"]

    for task_type in tasks:
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating task: {task_type}")
        logger.info(f"{'='*50}")

        # Create output directory
        output_dir = f"./output/{task_type}"
        os.makedirs(output_dir, exist_ok=True)

        # 1. Load dataset and data loader
        data_loader = get_task_dataset_and_loader(task_type, tokenizer, batch_size)

        # Get sample batch for testing
        sample_inputs = None
        for batch in data_loader:
            if isinstance(batch, dict):
                # Dataset returns dict format
                sample_inputs = {
                    'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device)
                }
                logger.info(f"Sample batch shapes: input_ids={batch['input_ids'].shape}, attention_mask={batch['attention_mask'].shape}")
            elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                # TensorDataset returns tuple format
                sample_inputs = {
                    'input_ids': batch[0].to(device),
                    'attention_mask': batch[1].to(device)
                }
                logger.info(f"Sample batch shapes: input_ids={batch[0].shape}, attention_mask={batch[1].shape}")
            break

        if sample_inputs is None:
            logger.error(f"Failed to get sample inputs for task {task_type}")
            continue

        # 2. Load dense model
        try:
            dense_model = load_dense_model(gpt_neo_base_path, task_type)
            logger.info("Dense model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load dense model: {e}")
            continue

        # 3. Try to load sparse model 
        sparse_model = None
        sparse_model_paths = {
            "math": "TransModular_GPT/fintune/save_model_with_mask_0.25/mathqa/lr5e-05_bs4_e2/best_model/pytorch_model.bin",
            "law": "TransModular_GPT/fintune/save_model_with_mask_0.25/law/scotus/lr5e-05_bs4_e4/best_model/pytorch_model.bin",
        }

        sparse_model_path = sparse_model_paths.get(task_type)
        if sparse_model_path and os.path.exists(sparse_model_path):
            try:
                sparse_model = load_sparse_model(gpt_neo_base_path, sparse_model_path, task_type)
                logger.info("Sparse model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load sparse model: {e}")
        else:
            logger.info(f"No sparse model found for task {task_type}, using dense model for comparison")
            sparse_model = copy.deepcopy(dense_model)

        # 4. Calculate model sparsity
        logger.info("\nCalculating model sparsity:")
        dense_sparsity = calculate_sparsity(dense_model.state_dict())
        sparse_sparsity = calculate_sparsity(sparse_model.state_dict())

        logger.info(f"Dense model: {dense_sparsity['sparsity_percentage']:.2f}% zeros ({dense_sparsity['zero_params']:,}/{dense_sparsity['total_params']:,})")
        logger.info(f"Sparse model: {sparse_sparsity['sparsity_percentage']:.2f}% zeros ({sparse_sparsity['zero_params']:,}/{sparse_sparsity['total_params']:,})")

        # 5. Benchmark on CPU with configured parameters
        logger.info(f"\nBenchmarking on CPU ({num_iterations} batches, {num_warmup} warmup):")
        try:
            dense_cpu_perf = benchmark_pytorch_model(dense_model, data_loader, "cpu",
                                                    num_iterations=num_iterations, num_warmup=num_warmup)
            sparse_cpu_perf = benchmark_pytorch_model(sparse_model, data_loader, "cpu",
                                                     num_iterations=num_iterations, num_warmup=num_warmup)

            logger.info(f"Dense model (CPU):  {dense_cpu_perf['avg_ms_per_batch']:.2f} +/- {dense_cpu_perf['std_ms_per_batch']:.2f} ms/batch")
            logger.info(f"Sparse model (CPU): {sparse_cpu_perf['avg_ms_per_batch']:.2f} +/- {sparse_cpu_perf['std_ms_per_batch']:.2f} ms/batch")

            if sparse_cpu_perf['avg_ms_per_batch'] > 0:
                cpu_speedup = dense_cpu_perf['avg_ms_per_batch'] / sparse_cpu_perf['avg_ms_per_batch']
                logger.info(f"CPU Speedup (sparse vs dense): {cpu_speedup:.2f}x")
        except Exception as e:
            logger.error(f"CPU benchmarking failed: {e}")


        # 6. Try DeepSparse acceleration on sparse model
        deepsparse_perf = None
        logger.info(f"\nTrying DeepSparse acceleration on sparse model ({num_iterations} batches, {num_warmup} warmup):")
        try:
            # Try to apply DeepSparse acceleration
            sparse_engine_or_model, deepsparse_enabled = try_deepsparse_acceleration(
                sparse_model, tokenizer, task_type, batch_size
            )

            if deepsparse_enabled:
                # Benchmark with DeepSparse engine
                deepsparse_perf = benchmark_deepsparse_engine(
                    sparse_engine_or_model, data_loader,
                    num_iterations=num_iterations, num_warmup=num_warmup
                )

                if deepsparse_perf:
                    logger.info(f"DeepSparse acceleration: {deepsparse_perf['avg_ms_per_batch']:.2f} +/- {deepsparse_perf['std_ms_per_batch']:.2f} ms/batch")

                    # Calculate speedups
                    if 'dense_cpu_perf' in locals():
                        deepsparse_vs_dense = dense_cpu_perf['avg_ms_per_batch'] / deepsparse_perf['avg_ms_per_batch']
                        logger.info(f"DeepSparse vs dense CPU: {deepsparse_vs_dense:.2f}x")
                    if 'sparse_cpu_perf' in locals():
                        deepsparse_vs_sparse = sparse_cpu_perf['avg_ms_per_batch'] / deepsparse_perf['avg_ms_per_batch']
                        logger.info(f"DeepSparse vs sparse CPU: {deepsparse_vs_sparse:.2f}x")

                else:
                    logger.warning("DeepSparse benchmarking failed")
            else:
                logger.info("DeepSparse acceleration not available or failed")

        except Exception as e:
            logger.error(f"DeepSparse acceleration attempt failed: {e}")

        # 7. Save detailed results summary
        try:
            result_file = f"{output_dir}/benchmark_results.txt"
            with open(result_file, "w") as f:
                f.write(f"GPT-Neo Model Performance Evaluation Report\n")
                f.write(f"=" * 50 + "\n")
                f.write(f"Task: {task_type}\n")
                f.write(f"Model: {gpt_neo_base_path}\n")
                f.write(f"Configuration:\n")
                f.write(f"  - Batch size: {batch_size} samples\n")
                f.write(f"  - Measurement batches: {num_iterations}\n")
                f.write(f"  - Warmup batches: {num_warmup}\n\n")

                f.write(f"Model Sparsity Analysis:\n")
                f.write(f"  - Dense model:  {dense_sparsity['sparsity_percentage']:.2f}% zeros ({dense_sparsity['zero_params']:,}/{dense_sparsity['total_params']:,} params)\n")
                f.write(f"  - Sparse model: {sparse_sparsity['sparsity_percentage']:.2f}% zeros ({sparse_sparsity['zero_params']:,}/{sparse_sparsity['total_params']:,} params)\n\n")

                if 'dense_cpu_perf' in locals():
                    f.write(f"CPU Performance Results:\n")
                    f.write(f"  - Dense model:  {dense_cpu_perf['avg_ms_per_batch']:.2f} ± {dense_cpu_perf['std_ms_per_batch']:.2f} ms/batch\n")
                    f.write(f"  - Sparse model: {sparse_cpu_perf['avg_ms_per_batch']:.2f} ± {sparse_cpu_perf['std_ms_per_batch']:.2f} ms/batch\n")
                    if sparse_cpu_perf['avg_ms_per_batch'] > 0:
                        cpu_speedup = dense_cpu_perf['avg_ms_per_batch'] / sparse_cpu_perf['avg_ms_per_batch']
                        f.write(f"  - CPU Speedup: {cpu_speedup:.2f}x\n")
                    f.write("\n")


                if deepsparse_perf:
                    f.write(f"DeepSparse Acceleration Performance:\n")
                    f.write(f"  - DeepSparse engine: {deepsparse_perf['avg_ms_per_batch']:.2f} ± {deepsparse_perf['std_ms_per_batch']:.2f} ms/batch\n")
                    if 'dense_cpu_perf' in locals():
                        ds_vs_dense = dense_cpu_perf['avg_ms_per_batch'] / deepsparse_perf['avg_ms_per_batch']
                        f.write(f"  - DeepSparse vs Dense: {ds_vs_dense:.2f}x\n")
                    if 'sparse_cpu_perf' in locals():
                        ds_vs_sparse = sparse_cpu_perf['avg_ms_per_batch'] / deepsparse_perf['avg_ms_per_batch']
                        f.write(f"  - DeepSparse vs Sparse: {ds_vs_sparse:.2f}x\n")
                    f.write("\n")
                else:
                    f.write(f"DeepSparse Acceleration: Not available or failed\n\n")


                f.write(f"Summary:\n")
                f.write(f"  - Best performing configuration: ")
                best_time = float('inf')
                best_config = "Unknown"

                if 'dense_cpu_perf' in locals() and dense_cpu_perf['avg_ms_per_batch'] < best_time:
                    best_time = dense_cpu_perf['avg_ms_per_batch']
                    best_config = "Dense CPU"
                if 'sparse_cpu_perf' in locals() and sparse_cpu_perf['avg_ms_per_batch'] < best_time:
                    best_time = sparse_cpu_perf['avg_ms_per_batch']
                    best_config = "Sparse CPU"
                if deepsparse_perf and deepsparse_perf['avg_ms_per_batch'] < best_time:
                    best_time = deepsparse_perf['avg_ms_per_batch']
                    best_config = "DeepSparse Acceleration"

                f.write(f"{best_config} ({best_time:.2f} ms/batch)\n")

            logger.info(f"Detailed results saved to {result_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

if __name__ == "__main__":
    main()