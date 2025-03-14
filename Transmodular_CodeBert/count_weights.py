from global_configs import GlobalConfigs
from transformers import (RobertaModel, RobertaTokenizer)
from global_configs import GlobalConfigs
from utils import load_init_module, load_init_module_sparse, load_init_module_sparse_lr
import torch
global_configs = GlobalConfigs(lang="python")
codebert_base = RobertaModel.from_pretrained(global_configs.codebert_base_path)
tokenizer = RobertaTokenizer.from_pretrained(global_configs.codebert_base_path)
tokenizer.do_lower_case = True
module_encoder = load_init_module_sparse_lr(codebert_base, "./data/module_python/lr_0.01_alpha_1.5_rank_64_lrlm_0.0005/result/pytorch_model.bin")
# module_encoder = load_init_module_sparse(codebert_base, "./data/module_python/lr_0.01_alpha_10.0/result/pytorch_model.bin")

for i, layer in enumerate(module_encoder.encoder.layer):
    total_elements = 0
    non_zero_elements = 0

    for param in layer.parameters():
        # 只考虑权重矩阵，忽略其它类型的参数
        if param.dim() > 1:
            total_elements += param.numel()
            non_zero_elements += torch.count_nonzero(param).item()

    # 计算非零比例
    non_zero_ratio = non_zero_elements / total_elements if total_elements > 0 else 0
    # print(f"Layer {i}: Non-zero ratio = {non_zero_ratio:.4f}")
    print(f"{non_zero_ratio:.4f}")
