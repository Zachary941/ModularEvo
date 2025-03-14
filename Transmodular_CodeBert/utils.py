import os
import torch
import copy
from mask_layer import init_mask_model, Binarization
from transformers import RobertaForMaskedLM
from transformers import AutoConfig, AutoModel

# reuse the initial module which is decomposed from the initial model,
# then fine-tune the initial module with on-demand reuse.
# the loaded module has MaskLinear layers.
def load_init_module(model, init_module_path):
    # if not os.path.exists(init_module_path):
    #     raise ValueError('Should modularize the pre-trained model first.')
    module_state = torch.load(f'{init_module_path}/pytorch_model.bin')
    # config_path = os.path.join(init_module_path, 'config.json')
    # model_path = os.path.join(init_module_path, 'model.safetensors')
    # training_args_path = os.path.join(init_module_path, 'training_args.bin')

    # config = AutoConfig.from_pretrained(config_path)
    # module = AutoModel.from_pretrained(model_path, config=config)

    # module_state = module.state_dict()
    print(f'load_init_module: {module_state.keys()}')


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
#####################################################################
def load_init_module_sparse_lr(model, module_path, prefix='roberta.'):
    if not os.path.exists(module_path):
        raise ValueError(f'{module_path} does not exist.')
    module_state = torch.load(module_path)
    model_state = model.state_dict()
    sparse_model_state = copy.deepcopy(model_state)

    same_k, mask_k, diff_k = [], [], []
    for k in model_state:
        tmp_k = f'{prefix}{k}'
        if tmp_k in module_state:
            same_k.append(k)
            # TODO modify the key to ``mask_A'' and ``_B''
            if f'{tmp_k}_mask_A' in module_state:
                mask_k.append(f'{tmp_k}_mask_A')
                mask_k.append(f'{tmp_k}_mask_B')
                init_weight = module_state[tmp_k]
                weight_mask = Binarization.apply\
                    (module_state[f'{tmp_k}_mask_A'] @ module_state[f'{tmp_k}_mask_B'])
                masked_weight = init_weight * weight_mask
                sparse_model_state[k] = masked_weight
            else:
                sparse_model_state[k] = module_state[tmp_k]
        else:
            diff_k.append(k)
    print(f'diff k: {diff_k}\n\n')
    model.load_state_dict(sparse_model_state)
    return model

# def load_init_module_compressed(model, module_path, prefix='roberta.'):
#     if not os.path.exists(module_path):
#         raise ValueError(f'{module_path} does not exist.')
#     module_state = torch.load(module_path)
#     model_state = model.state_dict()
#     sparse_model_state = copy.deepcopy(model_state)

#     same_k, mask_k, diff_k = [], [], []
#     for k in model_state:
#         tmp_k = f'{prefix}{k}'
#         if tmp_k in module_state:
#             same_k.append(k)
#             if f'{tmp_k}_pad' in 

#     pass
#####################################################################
def calculate_non_zero_percentage(model_state_dict):
    total_params = 0
    non_zero_params = 0

    for param_tensor in model_state_dict:
        total_params += torch.numel(model_state_dict[param_tensor])
        non_zero_params += torch.count_nonzero(model_state_dict[param_tensor])
        # print(f'{param_tensor}LAYER NON ZERO :{non_zero_params/total_params}')
    percentage = (non_zero_params / total_params) * 100
    return percentage
# reuse the initial module, and use MaskLinear layers to get sparse Linear layers.

def new_load_init_module(model, module_path):
    prefix='roberta.'
    module_state = torch.load(f'{module_path}/pytorch_model_try.bin')
    # print(f'module 111 : {module_state.keys()}')
    #根据mask的值，对model进行mask
    model_state = model.state_dict()
    # print(f'model 222 : {model_state.keys()}')
    # for k in module_state:
    #     print(f'module 111 : {k}')
    same_k, mask_k, diff_k = [], [], []
    for k in model_state:
    #     print(f'{k}_mask')
    #     print(f'module 222 : {k}')
        if f'roberta.{k}' in module_state:
            same_k.append(k)
            if f'roberta.{k}_mask' in module_state:
                mask = module_state[f'roberta.{k}_mask']
                bin_mask = (mask > 0).float()
                bin_mask = bin_mask.to(model_state[k].device)
                model_state[k] = model_state[k] * bin_mask
            else:
                model_state[k] = module_state[f'roberta.{k}']
        else:
            diff_k.append(k)

    # print(f'same k: {same_k}\n\n')
    # print(f'diff k: {diff_k}\n\n')

    model.load_state_dict(model_state)
    print(f'model_sparse: {calculate_non_zero_percentage(model_state)}')
    return model

def load_init_module_sparse(model, module_path, prefix='roberta.'):
    # if not os.path.exists(module_path):
    #     raise ValueError(f'{module_path} does not exist.')
    # # 这行就行
    module_state = torch.load(module_path)
    # module_path ='data/module_java/lr_0.01_alpha_10.0_ne_2_wrr_7.24/result/'
    # config_path = os.path.join(module_path, 'config.json')
    # model_path = os.path.join(module_path, 'model.safetensors')
    # training_args_path = os.path.join(module_path, 'training_args.bin')

    # config = AutoConfig.from_pretrained(config_path)
    # module = AutoModel.from_pretrained(model_path, config=config)
    model_state = model.state_dict()
    sparse_model_state = copy.deepcopy(model_state)

    same_k, mask_k, diff_k = [], [], []
    for k in model_state:
        tmp_k = f'{prefix}{k}'
        if tmp_k in module_state:
            same_k.append(k)
            # TODO modify the key to ``mask_A'' and ``_B''
            if f'{tmp_k}_mask' in module_state:
                mask_k.append(f'{tmp_k}_mask')
                init_weight = module_state[tmp_k]
                weight_mask = Binarization.apply(module_state[f'{tmp_k}_mask'])
                masked_weight = init_weight * weight_mask

                total_params += torch.numel(masked_weight)
                non_zero_params += torch.count_nonzero(masked_weight)
                print(f'{k}LAYER NON ZERO :{non_zero_params/total_params}')


                sparse_model_state[k] = masked_weight
            else:
                sparse_model_state[k] = module_state[tmp_k]
        else:
            diff_k.append(k)
    print(f'same k: {same_k}\n\n')
    print(f'diff k: {diff_k}\n\n')
    model.load_state_dict(sparse_model_state)
    return model


# load the on-demand reused module which is generated by on-demand reusing the initial module on the target task.
# use MaskLinear layers to get sparse Linear layers.
# the reused module's embedding layer has fewer embeddings than the initial module.
# So, here needs to process the embedding layer of `model`.
# Note, this only returns the encoder part.
def load_reused_module_sparse(model, module_path, prefix='encoder.'):
    if not os.path.exists(module_path):
        raise ValueError(f'{module_path} does not exist.')
    module_state = torch.load(module_path)  # the reused module contains the encoder and classifier.
    model_state = model.state_dict()  # the model corresponds to the encoder.
    sparse_model_state = copy.deepcopy(model_state)

    same_k, mask_k, diff_k = [], [], []
    for k in model_state:
        tmp_k = f'{prefix}{k}'
        if tmp_k in module_state:
            same_k.append(k)
            if f'{tmp_k}_mask' in module_state:
                mask_k.append(f'{tmp_k}_mask')
                init_weight = module_state[tmp_k]
                weight_mask = Binarization.apply(module_state[f'{tmp_k}_mask'])
                masked_weight = init_weight * weight_mask
                sparse_model_state[k] = masked_weight
            else:
                sparse_model_state[k] = module_state[tmp_k]
        else:
            diff_k.append(k)
    # print(f'same k: {same_k}\n\n')
    # print(f'mask k: {mask_k}\n\n')
    print(f'diff k: {diff_k}\n\n')

    model.embeddings.word_embeddings.weight.data = module_state[f'{prefix}embeddings.word_embeddings.weight']
    model.embeddings.word_embeddings.num_embeddings = len(module_state[f'{prefix}embeddings.word_embeddings.weight'])

    model.load_state_dict(sparse_model_state)
    return model


# load the reused model which is generated by fine-tuning the initial trained model on the target task.
# NOTE: this only returns the encoder part
def load_reused_model(model, model_path, prefix='encoder.'):
    if not os.path.exists(model_path):
        raise ValueError(f'{model_path} does not exist.')
    reused_model_state = torch.load(model_path)  # the reused model contains the encoder and classifier.
    model_state = model.state_dict()  # the model corresponds to the encoder.
    encoder_state = copy.deepcopy(model_state)

    same_k, mask_k, diff_k = [], [], []
    for k in model_state:
        tmp_k = f'{prefix}{k}'
        if tmp_k in reused_model_state:
            same_k.append(k)
            encoder_state[k] = reused_model_state[tmp_k]
        else:
            diff_k.append(k)
    print(f'diff k: {diff_k}\n\n')
    model.load_state_dict(encoder_state)
    return model

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
    # compared to codebert_base, codebert_mlm has an additional 'lm_head' as the output layer.
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
