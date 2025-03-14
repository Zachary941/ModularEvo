from hmac import new
import torch
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from task_eval.law_eval import evaluate_law_model,GPTNeoWithClassificationHead

def compute_sparsity(model_dict1, model_dict2, threshold=1e-6, plot=False, save_dir=None):
    """
    Calculate the sparsity between two model state dictionaries after subtraction
    
    Parameters:
        model_dict1 (OrderedDict): State dictionary of the first model
        model_dict2 (OrderedDict): State dictionary of the second model
        threshold (float): Threshold below which differences are considered zero
        plot (bool): Whether to plot sparsity heatmap
        save_dir (str): Directory to save plots if provided
        
    Returns:
        tuple: (overall_sparsity, dictionary of layer-wise sparsity)
    """

    # Check if dictionaries have common keys
    keys1 = set(model_dict1.keys())
    keys2 = set(model_dict2.keys())
    common_keys = keys1.intersection(keys2)
    
    # Exclude keys containing "classification"
    common_keys = {key for key in common_keys if "classification" not in key and "bias" not in key}
    
    if not common_keys:
        raise ValueError("The two model dictionaries have no common keys (excluding classification keys)")
    
    total_params = 0
    total_zeros = 0
    layer_sparsity = {}
    
    # Calculate differences and sparsity for each common key
    for key in common_keys:
        # Ensure parameters are tensors
        if not isinstance(model_dict1[key], torch.Tensor) or not isinstance(model_dict2[key], torch.Tensor):
            continue
            
        # Calculate parameter differences
        param_diff = model_dict1[key] - model_dict2[key]
        
        # Count elements close to zero
        num_params = param_diff.numel()
        num_zeros = torch.sum(torch.abs(param_diff)< 1e-6).item()
        
        # Calculate sparsity for this layer
        sparsity = num_zeros / num_params if num_params > 0 else 0
        layer_sparsity[key] = sparsity
        
        # Accumulate total parameters and zero elements
        total_params += num_params
        total_zeros += num_zeros
    
    # Calculate overall sparsity
    overall_sparsity = total_zeros / total_params if total_params > 0 else 0
    
    return overall_sparsity, layer_sparsity

def count_zero_weights(model_dict):
    """
    Count the number of zero weights in a model state dictionary
    
    Parameters:
        model_dict (OrderedDict): State dictionary of the model
        
    Returns:
        int: Total number of zero weights
        float: Proportion of zero weights
    """
    if not isinstance(model_dict, (dict, OrderedDict)):
        raise TypeError("Input must be a model state dictionary")
    
    total_params = 0
    total_zeros = 0
    
    for key, param in model_dict.items():
        if isinstance(param, torch.Tensor):
            num_params = param.numel()
            num_zeros = torch.sum(param ==0.0).item()
            
            total_params += num_params
            total_zeros += num_zeros
    
    zero_weight_proportion = total_zeros / total_params if total_params > 0 else 0
    return total_zeros, zero_weight_proportion

if __name__ == "__main__":
    model_name_or_path = "/home/LAB/longwr/new_SeaM/TransModular_GPT/data/gpt-neo-125m/"
    # model_math_state_dict = torch.load("/home/LAB/longwr/new_SeaM/TransModular_GPT/fintune/save_model/mathqa/lr5e-05_bs4_e2/best_model/pytorch_model.bin",map_location=torch.device('cuda'))
    # model_law_state_dict = torch.load("/home/LAB/longwr/new_SeaM/TransModular_GPT/fintune/save_model/law/scotus/lr5e-05_bs4_e4/result/pytorch_model.bin",map_location=torch.device('cuda'))
    model_math_state_dict = torch.load("/home/LAB/longwr/new_SeaM/TransModular_GPT/fintune/save_model_with_mask_0.25/mathqa/lr5e-05_bs4_e2/best_model/pytorch_model.bin",map_location=torch.device('cuda'))
    model_law_state_dict = torch.load("/home/LAB/longwr/new_SeaM/TransModular_GPT/fintune/save_model_with_mask_0.25/law/scotus/lr5e-05_bs4_e4/best_model/pytorch_model.bin",map_location=torch.device('cuda'))
    
    
    model_name_or_path = "/home/LAB/longwr/new_SeaM/TransModular_GPT/data/gpt-neo-125m/"
    merged_model = GPTNeoWithClassificationHead(model_name_or_path, num_classes=1).to('cuda')
    base_state_dict = merged_model.state_dict()
    new_state_dict = OrderedDict()
    for key in base_state_dict.keys():
        if key in model_math_state_dict:
            new_state_dict[key] = model_math_state_dict[key]-base_state_dict[key]
            print(key,new_state_dict[key].size())

    # Count zero weights in math model
    math_zero_weights, math_zero_proportion = count_zero_weights(new_state_dict)
    print(f"Math model zero weights: {math_zero_weights}")
    print(f"Math model zero weight proportion: {math_zero_proportion:.4f}")
    # print(f"diff:{compute_sparsity(model_math_state_dict, base_state_dict)}")

    new_state_dict = OrderedDict()
    for key in base_state_dict.keys():
        new_state_dict[key] = model_law_state_dict[key]-base_state_dict[key]
    # Count zero weights in law model
    law_zero_weights, law_zero_proportion = count_zero_weights(new_state_dict)
    print(f"Law model zero weights: {law_zero_weights}")
    print(f"Law model zero weight proportion: {law_zero_proportion:.4f}")
    # print(f"diff:{compute_sparsity(model_law_state_dict, base_state_dict)}")

# if __name__ == "__main__":
#     mask_rate=0.25
#     if mask_rate == 0.25:
#         module_state = torch.load("/home/LAB/longwr/new_SeaM/TransModular_GPT/data/module_law/lr_0.005_alpha_10.0_bs_4_time_20250227_104430/model_wrr_0.25/pytorch_model.bin")
#     elif mask_rate == 0.5:
#         module_state = torch.load("/home/LAB/longwr/new_SeaM/TransModular_GPT/data/module_law/lr_0.005_alpha_10.0_bs_4_time_20250227_104430/model_wrr_0.50/pytorch_model.bin")
#     elif mask_rate == 0.75:
#         module_state = torch.load("/home/LAB/longwr/new_SeaM/TransModular_GPT/data/module_law/lr_0.005_alpha_10.0_bs_4_time_20250227_104430/model_wrr_0.75/pytorch_model.bin")
    
#     model_name_or_path = "/home/LAB/longwr/new_SeaM/TransModular_GPT/data/gpt-neo-125m/"
#     # model_math_state_dict = torch.load("/home/LAB/longwr/new_SeaM/TransModular_GPT/fintune/save_model_with_mask/mathqa/lr5e-05_bs4_e2/best_model/pytorch_model.bin",map_location=torch.device('cuda'))
#     model_math_state_dict = torch.load("/home/LAB/longwr/new_SeaM/TransModular_GPT/fintune/save_model/mathqa/lr5e-05_bs4_e2/best_model/pytorch_model.bin")
#     model_law_state_dict = torch.load("/home/LAB/longwr/new_SeaM/TransModular_GPT/fintune/save_model/law/scotus/lr5e-05_bs4_e4/result/pytorch_model.bin")
#     total_params_count = 0
#     masked_params_count = 0
#     new_state_dict={}
#     for name, param in model_math_state_dict:
#         # total_params_count += param.numel()
#         modify_name = name.replace("base_model.", "")
#         if f"{modify_name}_mask" in module_state:
#             total_params_count += param.numel()
#             mask = module_state[f'{modify_name}_mask']
#             bin_mask = (mask > 0).float()
#             masked_params_count += bin_mask.sum().item()
#             new_state_dict[name]= param.data *bin_mask
    
#     merged_model = GPTNeoWithClassificationHead(new_state_dict, num_classes=1)
#     merged_model = merged_model.to('cuda')
#     print(count_zero_weights(merged_model.state_dict()))
    # A,B=compute_sparsity(model_law_state_dict, merged_model.state_dict(), plot=False, save_dir="sparsity_plots")
    # print(A)
    # print(B)
