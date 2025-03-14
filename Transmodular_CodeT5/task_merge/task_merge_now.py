from __future__ import absolute_import
import os
import sys
import pickle
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
import datetime
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from collections import OrderedDict
import copy
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer,
                          BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)
import multiprocessing
from task_eval.clone_eval import do_clone_test
from task_eval.gen_eval import do_gen_test

sys.path.append('../../')
sys.path.append('../')
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
                 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
                 'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer)}
from evaluator import smooth_bleu
from evaluator.CodeBLEU import calc_code_bleu
from evaluator.bleu import _bleu
from utils import read_examples, convert_examples_to_features, calc_stats
from configs import add_args, set_seed, set_dist
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
cpu_cont = multiprocessing.cpu_count()
pool = multiprocessing.Pool(cpu_cont)

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def write_summary(alphas, bleus, output_file='summary.txt'):
    avg_bleu = sum(bleus) / len(bleus)

    if not os.path.exists(output_file):
        with open(output_file, 'w') as f:
            header = []
            header.extend([f"Alpha{i+1}_BLEU4" for i in range(len(alphas))])
            header.extend([f"Task{i+1}_BLEU4" for i in range(len(bleus))])
            header.append("Avg_BLEU4")
            f.write("\t".join(header) + "\n")
            f.write("-" * 80 + "\n")

    with open(output_file, 'a') as f:
        row = []
        row.extend([f"{alpha:.4f}" for alpha in alphas])
        row.extend([f"{bleu:.4f}" for bleu in bleus])
        row.append(f"{avg_bleu:.4f}")
        f.write("\t".join(row) + "\n")
    
    return avg_bleu


def calculate_non_zero_percentage(model_state_dict):
    total_params = 0
    non_zero_params = 0

    for param_tensor in model_state_dict:
        total_params += torch.numel(model_state_dict[param_tensor])
        non_zero_params += torch.count_nonzero(model_state_dict[param_tensor])
        # print(f'{param_tensor}LAYER NON ZERO :{non_zero_params/total_params}')
    percentage = (non_zero_params / total_params) * 100
    # print(f'NON ZERO :{percentage}')    
    return percentage


def add_tv(tv1,tv2,scaling):
    """Add two task vectors together."""
    with torch.no_grad():
        new_vector = {}
        for key in tv1:
            if key not in tv2:
                print(f'Warning, key {key} is not present in both task vectors.scaling1:{scaling[0]},scaling2:{scaling[1]}')
                continue
            new_vector[key] = scaling[0] * tv1[key] + scaling[1] * tv2[key]
        # print(f'{scaling[0]}*{tv1[key]}+{scaling[1]}*{tv2[key]}\n')
    return new_vector

def add_tvs(tvs, alphas):
    """Add multiple task vectors with their corresponding scaling factors."""
    if len(tvs) != len(alphas):
        raise ValueError(f"task num :({len(tvs)}) vector num : ({len(alphas)}) error!")
    
    with torch.no_grad():
        new_vector = {}
        for key in tvs[0]:
            if not all(key in tv for tv in tvs):
                print(f'Warning, key {key} is not present in all task vectors')
                continue
            new_vector[key] = sum(alpha * tv[key] for tv, alpha in zip(tvs, alphas))
    return new_vector

def get_vector(pretrained_state_dict,finetuned_state_dict):
    tv1 = OrderedDict()
    total_params = 0
    non_zero_params = 0
    device='cuda'
    for key in pretrained_state_dict:
        if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
            continue
        # if 'encoder.encoder' in key:
        #     tv1[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
        #     total_params += torch.numel(tv1[key])
        #     non_zero_params += torch.count_nonzero(tv1[key])
        #     # print(f'{param_tensor}LAYER NON ZERO :{non_zero_params/total_params}')
        #     percentage = (non_zero_params / total_params) * 100
        #     print(f'get vector {key} :{percentage}')
        #     # print('GET VECTOR:',key).
        pretrained_tensor = pretrained_state_dict[key].to(device)
        finetuned_tensor = finetuned_state_dict[key].to(device)
        tv1[key] = finetuned_tensor - pretrained_tensor
        total_params += torch.numel(tv1[key])
        non_zero_params += torch.count_nonzero(tv1[key])      
    print(f'percentage :{non_zero_params/total_params},non_zero_params:{non_zero_params},total_params:{total_params}')
        # print(f'name {key} :111{finetuned_state_dict[key]}  222{pretrained_state_dict[key]} 333{tv1[key]}')
    return tv1

def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6))


def calculate_overlap_mask(task_vectors):

    overlap_mask = OrderedDict()
    for key in task_vectors[0].keys():
        overlap_mask[key] = sum((tv[key] != 0).float() for tv in task_vectors)
    return overlap_mask

def split_task_vectors(task_vectors, alphas):

    num_tasks = len(task_vectors)
    threshold = max(2, round(num_tasks * 2/3))
    overlap_mask = calculate_overlap_mask(task_vectors)
    # print(f'overlap_mask:{overlap_mask}')
    base_tv = OrderedDict()
    diff_tvs = [OrderedDict() for _ in task_vectors]
    
    for key in task_vectors[0].keys():
        base_mask = (overlap_mask[key] >= threshold).float()
        diff_mask = (overlap_mask[key] < threshold).float()
        
        weighted_sum = sum(tv[key] * alpha for tv, alpha in zip(task_vectors, alphas))
        base_tv[key] = weighted_sum * base_mask
        
        for i, tv in enumerate(task_vectors):
            diff_tvs[i][key] = tv[key] * diff_mask
    
    return base_tv, diff_tvs

def calculate_binary_overlap_statistics(tv1, tv2):
    overlap_count = 0
    total_weights = 0
    
    for key in tv1.keys():
        mask1 = (tv1[key] != 0).float()
        mask2 = (tv2[key] != 0).float()
        overlap = (mask1 * mask2).sum().item() 
        
        total = mask1.numel() 
        overlap_count += overlap
        total_weights += total
    
    overlap_ratio = (overlap_count / total_weights) * 100
    non_overlap_ratio = 100 - overlap_ratio
    
    print("\nBinary Overlap Statistics:")
    print(f"Overlapping weights: {overlap_ratio:.2f}% ({overlap_count} weights)")
    print(f"Non-overlapping weights: {non_overlap_ratio:.2f}% ({total_weights - overlap_count} weights)")
    print(f"Total weights: {total_weights}")
    
    return overlap_ratio, non_overlap_ratio

def auto_alpha_set(task_vectors):
    num_tasks = len(task_vectors)
    overlap_mask = calculate_overlap_mask(task_vectors)
    threshold = max(2, round(num_tasks * 2/3))
    
    base_tv = OrderedDict()
    diff_tvs = [OrderedDict() for _ in task_vectors]
    
    for key in task_vectors[0].keys():

        base_mask = (overlap_mask[key] >= threshold).float()
        diff_mask = (overlap_mask[key] < threshold).float()
        all_mask = (overlap_mask[key] > 0).float()
        overlap_count = overlap_mask[key]
        # dynamic_alpha = 1.0 / overlap_count.clamp(min=1.0)  
        dynamic_alpha = torch.where(overlap_count > 0, 
                          1.0 / overlap_count, 
                          torch.ones_like(overlap_count))
        weighted_sum = sum(tv[key] * dynamic_alpha for tv in task_vectors)
        # base_tv[key] = weighted_sum * base_mask
        base_tv[key] = weighted_sum * all_mask
        # for i, tv in enumerate(task_vectors):
        #     diff_tvs[i][key] = tv[key] * diff_mask
    # return base_tv, diff_tvs
    return base_tv

def model_merge(args):
    bleus = []
    # load all model
    config_class, model_class, tokenizer_class = MODEL_CLASSES['codet5']
    model_name_or_path = 'TransModular_CodeT5/data/pretrain_model/codet5_small/'
    config = config_class.from_pretrained(model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
    model = model_class.from_pretrained(model_name_or_path)
    logger.info("Finish loading model [%s] from %s", get_model_size(model), model_name_or_path)
    base_state_dict = model.state_dict()

    # load model1
    model_ft1 = copy.deepcopy(model)
    model_ft1.to('cuda')
    # model_state1 = torch.load("TransModular_CodeT5/sh/saved_models/summarize/python/codet5_small_all_lr5_bs64_src256_trg128_pat3_e10/checkpoint-best-bleu/pytorch_model.bin")
    model_state1 = torch.load("TransModular_CodeT5/sh/saved_models_lota/summarize/python/codet5_small_all_lr5_bs64_src256_trg128_pat2_e10_wrr0.25/checkpoint-best-bleu/pytorch_model.bin")
    model_ft1.load_state_dict(model_state1)

    # load model2
    model_ft2 = copy.deepcopy(model)
    # model_state2 = torch.load("TransModular_CodeT5/sh/saved_models/concode/codet5_small_all_lr10_bs32_src320_trg150_pat3_e30/checkpoint-best-bleu/pytorch_model.bin")
    model_state2 = torch.load("TransModular_CodeT5/sh/saved_models_lota/concode/java/codet5_small_all_lr10_bs32_src320_trg150_pat3_e10_wrr0.25/checkpoint-best-bleu/pytorch_model.bin")
    model_ft2.load_state_dict(model_state2)
    model_ft2.to('cuda')

    # generate task_vector
    tv1 = get_vector(base_state_dict,model_state1)
    tv2 = get_vector(base_state_dict,model_state2)
    # tv3 = get_vector(base_state_dict,model_state3)
    # tv4 = get_vector(base_state_dict,model_state4)
    tvs=[tv1,tv2]
    tv_sum= add_tvs(tvs,[args.alpha1,args.alpha2])

    # # apply task_vector to model1
    print(f'tv1 wrr : {calculate_non_zero_percentage(tv1)}')
    print(f'tv2 wrr : {calculate_non_zero_percentage(tv2)}')
    print(f'tv_sum wrr : {calculate_non_zero_percentage(tv_sum)}')

    merge_model = copy.deepcopy(model)
    merge_model.to('cuda')

    new_state_dict = OrderedDict()
    for key in model_ft1.state_dict():
        # print(key)
        new_state_dict[key] = base_state_dict[key].to('cuda') + tv_sum[key].to('cuda')

    # model_ft1.load_state_dict(new_state_dict,strict=False)
    # model_ft2.load_state_dict(new_state_dict,strict=False)
    merge_model.load_state_dict(new_state_dict)

    print('======================python====================')
    result,bleu=do_gen_test(merge_model,tokenizer,task='summarize',sub_task='python',type='model_merge')
    print(result)
    bleus.append(bleu)
    print('======================java====================')
    result,bleu=do_gen_test(merge_model,tokenizer,task='concode',sub_task='java',type='model_merge')
    print(result)
    bleus.append(bleu)

    avg_bleu = write_summary(alphas=[args.alpha1,args.alpha2] ,bleus=bleus,output_file=args.log)
    print(f'Average BLEU: {avg_bleu:.4f}')

def module_merge(args):
    bleus = []
    # load all model
    config_class, model_class, tokenizer_class = MODEL_CLASSES['codet5']
    model_name_or_path = 'TransModular_CodeT5/data/pretrain_model/codet5_small/'
    config = config_class.from_pretrained(model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
    model = model_class.from_pretrained(model_name_or_path)
    logger.info("Finish loading model [%s] from %s", get_model_size(model), model_name_or_path)
    base_state_dict = model.state_dict()
 
    # load module1
    # module_path1="TransModular_CodeT5/sh/saved_models_module/summarize/python/codet5_small_all_lr5_bs64_src256_trg128_pat3_e10_wrr0.25/checkpoint-best-bleu/pytorch_model.bin"
    module_path1 = "TransModular_CodeT5/sh/saved_models_module/summarize/python/codet5_small_all_lr5_bs64_src256_trg128_pat2_e5_wrr0.25/checkpoint-best-bleu/pytorch_model.bin"
    module_1 = copy.deepcopy(model)
    module_1.load_state_dict(torch.load(module_path1))
    module_1.to('cuda')
    print(f'module1:{calculate_non_zero_percentage(module_1.state_dict())}')
    # print('======================code_clone====================')
    # result=evaluate_clone(module_1,tokenizer,"Clone_detection_BigCloneBench_2/dataset/test.txt",args.output_dir)
    # print(result)

    # load module2

    module_path2="TransModular_CodeT5/sh/saved_models_module/concode/java/codet5_small_all_lr10_bs32_src320_trg150_pat3_e30_wrr0.25/checkpoint-best-bleu/pytorch_model.bin"
    module_2 = copy.deepcopy(model)
    module_2.load_state_dict(torch.load(module_path2))
    module_2.to('cuda')
    print(f'module2:{calculate_non_zero_percentage(module_2.state_dict())}')

    # generate task_vector
    finetuned_state_dict1 = module_1.state_dict()
    finetuned_state_dict2 = module_2.state_dict()

    tv1 = OrderedDict()
    tv2 = OrderedDict()
    tv1 = get_vector(base_state_dict,finetuned_state_dict1)
    tv2 = get_vector(base_state_dict,finetuned_state_dict2)
    tv_sum= add_tv(tv1,tv2,[args.alpha1,args.alpha2])
    # tv_sum = auto_alpha_set([tv1,tv2])
    print(f'tv1 wrr : {calculate_non_zero_percentage(tv1)}')  
    print(f'tv2 wrr : {calculate_non_zero_percentage(tv2)}')
    print(f'tv_sum wrr : {calculate_non_zero_percentage(tv_sum)}')  

    merge_model = copy.deepcopy(model)
    merge_model.to('cuda')
    
    # apply task_vector 
    new_state_dict = OrderedDict()
    for key in module_1.state_dict():
        # print(key)
        new_state_dict[key] = base_state_dict[key].to('cuda') + tv_sum[key].to('cuda')
    # module_1.load_state_dict(new_state_dict,strict=False)
    # module_2.load_state_dict(new_state_dict,strict=False)
    merge_model.load_state_dict(new_state_dict)

    print('======================python====================')
    result,bleu1=do_gen_test(merge_model,tokenizer,task='summarize',sub_task='python',type='module_merge')
    print(result)
    bleus.append(bleu1)
    print('======================java====================')
    result,bleu2=do_gen_test(merge_model,tokenizer,task='concode',sub_task='java',type='module_merge')
    print(result)
    bleus.append(bleu2)
    avg_bleu = write_summary(alphas=[args.alpha1,args.alpha2] ,bleus=bleus,output_file=args.log)
    print(f'Average BLEU: {avg_bleu:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--merge_type", default=1, type=int, required=True,
                        help="1:model merge, 2:module merge")
    parser.add_argument("--alpha1", default=0, type=float)    
    parser.add_argument("--alpha2", default=1, type=float)    
    parser.add_argument("--log", type=str)   
    args = parser.parse_args()
    # model_merge(args)
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'./log/summarize_{current_time}.txt'
    args.log=output_file

    # if args.merge_type == 1:
    #     # old_stdout = sys.stdout
    #     # sys.stdout = output_file1
    #     model_merge(args)

    # elif args.merge_type == 2:
    #     # old_stdout = sys.stdout
    #     # sys.stdout = output_file2
    #     module_merge(args)

    # output_file1 = open('code-text/code/output1.txt', 'w')
    # output_file2 = open('code-text/code/output2.txt', 'w')

    for alpha1 in [round(x * 0.1, 1) for x in range(3, 12)]:
        for alpha2 in [round(x * 0.1, 1) for x in range(3 , 11)]:
    # for alpha1 in [0,1]:
    #     for alpha2 in [0,1]:
            args.alpha1 = alpha1
            args.alpha2 = alpha2
            print(f'Running with alpha1={args.alpha1}, alpha2={args.alpha2}')
            if args.merge_type == 1:
                # old_stdout = sys.stdout
                # sys.stdout = output_file1
                model_merge(args)
            elif args.merge_type == 2:
                # old_stdout = sys.stdout
                # sys.stdout = output_file2
                module_merge(args)
            # elif args.merge_type == 3:
            #     # old_stdout = sys.stdout
            #     # sys.stdout = output_file2
            #     new_module_merge(args)
    pool.close()
    pool.join()

