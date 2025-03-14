import os
import torch
import copy
from mask_layer import init_mask_model
from transformers import RobertaForMaskedLM


# reuse a module, then fine-tune with on-demand reuse.
def load_module(model, pre_module_path):
    if not os.path.exists(pre_module_path):
        raise ValueError('Should modularize the pre-trained model first.')
    module_state = torch.load(f'{pre_module_path}/pytorch_model.bin')

    masked_model = init_mask_model(model, no_mask=['pooler'], is_binary=False)
    masked_model_state = masked_model.state_dict()
    new_masked_model_state = copy.deepcopy(masked_model_state)
    same_k = []
    diff_k = []
    for k in masked_model_state:
        tmp_k = f'roberta.{k}'
        if tmp_k in module_state:
            same_k.append(k)
            new_masked_model_state[k] = module_state[tmp_k]
        else:
            diff_k.append(k)
    print(f'diff k: {diff_k}\n\n')
    masked_model.load_state_dict(new_masked_model_state)
    return masked_model


# reuse a model, then fine-tune with on-demand reuse
def load_masked_model_mlm(model, global_configs):
    codebert_mlm = RobertaForMaskedLM.from_pretrained(global_configs.pre_trained_model)
    codebert_mlm_state = codebert_mlm.state_dict()

    model_state = model.state_dict()
    new_model_state = copy.deepcopy(model_state)
    same_k = []
    diff_k = []
    for k in model_state:
        tmp_k = f'roberta.{k}'
        if tmp_k in codebert_mlm_state:
            same_k.append(k)
            new_model_state[k] = codebert_mlm_state[tmp_k]
        else:
            diff_k.append(k)
    print(f'diff k: {diff_k}\n\n')
    model.load_state_dict(new_model_state)
    masked_model = init_mask_model(model, no_mask=['pooler'], is_binary=False)

    return masked_model


# reuse a model, then standard fine-tune WITHOUT on-demand reuse.
def load_model_mlm(model, global_configs):
    codebert_mlm = RobertaForMaskedLM.from_pretrained(global_configs.pre_trained_model)
    codebert_mlm_state = codebert_mlm.state_dict()

    model_state = model.state_dict()
    new_model_state = copy.deepcopy(model_state)
    same_k = []
    diff_k = []
    for k in model_state:
        tmp_k = f'roberta.{k}'
        if tmp_k in codebert_mlm_state:
            same_k.append(k)
            new_model_state[k] = codebert_mlm_state[tmp_k]
        else:
            diff_k.append(k)
    print(f'diff k: {diff_k}\n\n')
    model.load_state_dict(new_model_state)
    return model
