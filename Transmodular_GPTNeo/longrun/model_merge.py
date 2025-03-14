import math
import os
from pyexpat import model
import sys
import time
import torch
import logging
import argparse
import numpy as np
from pathlib import Path
from transformers import GPT2Tokenizer
from typing import Dict, List, Tuple
# Add project path
sys.path.append('../../')
sys.path.append('../')

from task_eval.law_eval import GPTNeoWithClassificationHead
from task_merge.merge_methods.merging_methods import MergingMethod
from task_eval.law_eval import evaluate_law_model,GPTNeoWithClassificationHead
from task_eval.math_eval import evaluate_math_model
from task_merge.merge_utils.merge_utils import set_random_seed
from pre_data import load_dataset_new, set_seed
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    log_dir = "TransModular_GPT/task_merge/longrun/logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(os.path.join(log_dir, f"model_merge_{timestamp}.log"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    return logging.getLogger(__name__)

def merge_models(model1, model2, stage, model_base_path, 
                num_classes1=25, num_classes2=13, method="task_arithmetic", 
                coef1=1.0, coef2=1.0, exclude_patterns=None):

    if exclude_patterns is None:
        exclude_patterns = [".*classification.*"]
        
    logger.info(f"Merging models using method: {method}")
    logger.info(f"Model 1, coefficient: {coef1}")
    logger.info(f"Model 2, coefficient: {coef2}")
    

    # Prepare merged model
    merged_model = GPTNeoWithClassificationHead(model_base_path, num_classes=1)
    
    # Initialize merging method
    merging_method = MergingMethod(merging_method_name=method)
    if method == "task_arithmetic":
        param_value_mask_rate=0
        mask_apply_method='task_arithmetic'
        use_weight_rescale=False
    elif method == "ties_merging":
        param_value_mask_rate=0.75
    elif method == "mask_merging":
        mask_apply_method='task_arithmetic'
        use_weight_rescale=True
    else:
        raise ValueError(f"Invalid merging method: {method}")       

    # Merge models
    logger.info("Performing model merge...")
    merged_params = merging_method.get_merged_model(
                                                merged_model=merged_model,
                                                models_to_merge=[model1, model2],
                                                exclude_param_names_regex=[".*classification.*"],
                                                scaling_coefficient=1,
                                                scaling_coefficients=[coef1, coef2],
                                                param_value_mask_rate=param_value_mask_rate,
                                                weight_mask_rates=[0.75 for _ in range(2)],
                                                use_weight_rescale=use_weight_rescale,
                                                mask_strategy='random',
                                                mask_apply_method=mask_apply_method,
                                                models_use_deepcopy=True
    )
    
    # # Save merged model
    # output_path = f"TransModular_GPT/task_merge/longrun/mask_fintune_{stage}/"
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # svae_params_path = output_path + f"merged_model_{stage}.bin"
    # torch.save(merged_params, svae_params_path)
    # logger.info(f"Merged model saved to {svae_params_path}")
    
    return merged_params
def parse_args():
    parser = argparse.ArgumentParser("Finetune task-specific model")
    parser.add_argument("--type", type=int, default=1)
    parser.add_argument("--alpha1", type=float, default=0.5)
    parser.add_argument("--alpha2", type=float, default=0.5)
    return parser.parse_args()

def merge1(args):
    model_name_or_path = "TransModular_GPT/data/gpt-neo-125m/"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    if args.type == 1:
        logger.info('======================math====================')
        model_math = GPTNeoWithClassificationHead(model_name_or_path, num_classes=25)
        model_math.load_state_dict(torch.load(""))

        logger.info('======================law====================')
        model_law = GPTNeoWithClassificationHead(model_name_or_path, num_classes=13)
        model_law.load_state_dict(torch.load(""))
        merged_params =merge_models(model_math, model_law, 0, model_name_or_path, num_classes1=25, num_classes2=13, method="task_arithmetic", coef1=args.alpha1, coef2=args.alpha2, exclude_patterns=None)
    else :
        logger.info('======================math====================')
        model_math = GPTNeoWithClassificationHead(model_name_or_path, num_classes=25)
        model_math.load_state_dict(torch.load(""))

        logger.info('======================law====================')
        model_law = GPTNeoWithClassificationHead(model_name_or_path, num_classes=13)
        model_law.load_state_dict(torch.load(""))
        merged_params =merge_models(model_math, model_law, 0, model_name_or_path, num_classes1=25, num_classes2=13, method="task_arithmetic", coef1=args.alpha1, coef2=args.alpha2, exclude_patterns=None)
    model_math.load_state_dict(merged_params,strict=False)
    model_law.load_state_dict(merged_params,strict=False)
    
    evaluate_math_model(model_math)
    evaluate_law_model(model_law)

    if args.type == 1:
        
        torch.save(model_math.state_dict(), "TransModular_GPT/task_merge/longrun/mask_fintune_0/model_merge_math0_full.bin")
        torch.save(model_law.state_dict(), "TransModular_GPT/task_merge/longrun/mask_fintune_0/model_merge_law0_full.bin")
    else:
        torch.save(model_math.state_dict(), "TransModular_GPT/task_merge/longrun/mask_fintune_0/model_merge_math0_mask.bin")
        torch.save(model_law.state_dict(), "TransModular_GPT/task_merge/longrun/mask_fintune_0/model_merge_law0_mask.bin")

def merge2(args):
    model_name_or_path = "TransModular_GPT/data/gpt-neo-125m/"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    print(args.type)
    print('======================')
    if args.type == 1:
        logger.info('======================math====================')
        model_math = GPTNeoWithClassificationHead(model_name_or_path, num_classes=25)
        # model_math.load_state_dict(torch.load("TransModular_GPT/task_merge/longrun/mask_fintune_0/mathqa_stage1_full/lr5e-05_bs8_e2_p3/best_model/pytorch_model.bin"))
        model_math.load_state_dict(torch.load("TransModular_GPT/task_merge/longrun/mask_fintune_0/mathqa_stage1_full/lr5e-05_bs8_e4_p20/checkpoint-5836/pytorch_model.bin"))
        logger.info('======================law====================')
        model_law = GPTNeoWithClassificationHead(model_name_or_path, num_classes=13)
        # model_law.load_state_dict(torch.load("TransModular_GPT/task_merge/longrun/mask_fintune_0/scotus_stage1_full/lr5e-05_bs8_e4_p4/best_model/pytorch_model.bin"))
        model_law.load_state_dict(torch.load("TransModular_GPT/task_merge/longrun/mask_fintune_0/scotus_stage1_full/lr5e-05_bs8_e4_p20/checkpoint-836/pytorch_model.bin"))
        merged_params =merge_models(model_math, model_law, 0, model_name_or_path, num_classes1=25, num_classes2=13, method="task_arithmetic", coef1=args.alpha1, coef2=args.alpha2, exclude_patterns=None)
    else :
        logger.info('======================math====================')
        model_math = GPTNeoWithClassificationHead(model_name_or_path, num_classes=25)
        model_math.load_state_dict(torch.load("TransModular_GPT/task_merge/longrun/mask_fintune_0/mathqa_stage1_mask/lr5e-05_bs8_e2_p8/best_model/pytorch_model.bin"))

        logger.info('======================law====================')
        model_law = GPTNeoWithClassificationHead(model_name_or_path, num_classes=13)
        model_law.load_state_dict(torch.load("TransModular_GPT/task_merge/longrun/mask_fintune_0/scotus_stage1_mask/lr5e-05_bs8_e6_p6/best_model/pytorch_model.bin"))
        merged_params =merge_models(model_math, model_law, 0, model_name_or_path, num_classes1=25, num_classes2=13, method="task_arithmetic", coef1=args.alpha1, coef2=args.alpha2, exclude_patterns=None)
    
    model_math.load_state_dict(merged_params,strict=False)
    model_law.load_state_dict(merged_params,strict=False)
    
    result_math=evaluate_math_model(model_math)

    result_law=evaluate_law_model(model_law)
    print('======================')
    print(result_math)
    print('======================')
    print(result_law)
    if args.type == 1:
        torch.save(model_math.state_dict(), "TransModular_GPT/task_merge/longrun/mask_fintune_0/model_merge_math1_full.bin")
        torch.save(model_law.state_dict(), "TransModular_GPT/task_merge/longrun/mask_fintune_0/model_merge_law1_full.bin")
    else:
        torch.save(model_math.state_dict(), "TransModular_GPT/task_merge/longrun/mask_fintune_0/model_merge_math1_mask.bin")
        torch.save(model_law.state_dict(), "TransModular_GPT/task_merge/longrun/mask_fintune_0/model_merge_law1_mask.bin")

if __name__ == "__main__":
    set_random_seed(42)
    logger = setup_logging()
    args = parse_args()
    merge2(args=args)
