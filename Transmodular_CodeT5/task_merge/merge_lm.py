import copy
import os
import sys
import argparse
from functools import partial
import time
import logging
import json
import torch
import numpy as np
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer,
                          BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)
from merge_utils.merge_utils import set_random_seed
from merge_methods.merging_methods import MergingMethod
from task_eval.gen_eval import do_gen_test

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
                 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
                 'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer)}

parser = argparse.ArgumentParser("Interface for merging PLMs on glue")
parser.add_argument("--language_model_name", type=str, default="codet5", help="name of the language model", choices=["bert-base-uncased", "roberta-base","codet5"])
parser.add_argument("--merging_method_name", type=str, default="average_merging", help="name of the method to merge models",
                    choices=["average_merging", "task_arithmetic",  "ties_merging", "mask_merging"])
parser.add_argument("--scaling_coefficient", type=float, default=1.0, help="scaling coefficient to merge the task vector")
parser.add_argument("--scaling_coefficients", type=list, default=[1.0, 1.0], help="scaling coefficients to merge the task vector")
parser.add_argument("--param_value_mask_rate", type=float, default=0.8, help="mask rate of the smallest-magnitude parameter values")
parser.add_argument("--weight_format", type=str, help="the format of weights to be masked", default="delta_weight", choices=["finetuned_weight", "delta_weight"])
parser.add_argument("--weight_mask_rate", type=float, default=0.1, help="weight mask rate")
parser.add_argument("--use_weight_rescale", action="store_true", default=False, help="whether to rescale the weight by 1 / (1 - weight_mask_rate)")
parser.add_argument("--mask_strategy", type=str, help="mask strategy", default="random", choices=["random", "magnitude"])
parser.add_argument("--mask_apply_method", type=str, default="average_merging", help="merging method that the mask strategy applies",
                    choices=["average_merging", "task_arithmetic",  "ties_merging"])
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--gpu", type=int, default=1, help="number of gpu to use")
parser.add_argument("--model_path1", type=str, help="model to merge 1")
parser.add_argument("--model_path2", type=str, help="model to merge 2")

parser.add_argument("--nums_fisher_examples", type=int, nargs="+", help="numbers of examples to compute fisher weights")
parser.add_argument("--fisher_scaling_coefficients", type=float, nargs="+", help="scaling coefficients to merge fisher weights")
parser.add_argument("--normalize_fisher_weight", action="store_true", default=False, help="whether to normalize fisher weights (L2 norm) or not")
parser.add_argument("--minimal_fisher_weight", type=float, default=1e-6, help="the minimal value in fisher weights, used for tackling the potential numerical issues")
parser.add_argument("--nums_regmean_examples", type=int, nargs="+", help="numbers of examples to compute regmean weights")
parser.add_argument("--reduce_non_diagonal_ratio", type=float, default=1.0, help="reduce non-diagonal elements in regmean weights by multiplying this scalar")


try:
    args = parser.parse_args()
    args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
except:
    parser.print_help()
    sys.exit()

def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6))

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

def get_merge_performance(args: argparse.Namespace, models_to_merge: list,  logger: logging.Logger,
                          merging_method: MergingMethod, tokenizer: transformers.AutoTokenizer):
    """
    get the performance of merging method named merging_method_name
    :param args: ArgumentParser, input argument parser
    :param models_to_merge: list, individual models that need to be merged
    :param logger: Logger, logger
    :param merging_method: MergingMethod, the mering method
    :param tokenizer: AutoTokenizer, tokenizer
    :return:
    """
    logger.info(f"configuration is {args}")
    bleus =[]
    config_class, model_class, tokenizer_class = MODEL_CLASSES['codet5']
    model_name_or_path = 'TransModular_CodeT5/data/pretrain_model/codet5_small/'
    config = config_class.from_pretrained(model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
    merged_model = model_class.from_pretrained(model_name_or_path)
    logger.info("Finish loading model [%s] from %s", get_model_size(merged_model), model_name_or_path)
    base_state_dict = merged_model.state_dict()
    # set random seed to guarantee reproducibility
    set_random_seed(seed=0)
    # exclude parameter whose name matches "classifier"
    merged_model = merging_method.get_merged_model(merged_model=merged_model,
                                                   models_to_merge=models_to_merge,
                                                   exclude_param_names_regex=[".*classifier.*"],
                                                   scaling_coefficient=args.scaling_coefficient,
                                                   scaling_coefficients=args.scaling_coefficients,
                                                   nums_fisher_examples=args.nums_fisher_examples,
                                                   fisher_scaling_coefficients=args.fisher_scaling_coefficients,
                                                   normalize_fisher_weight=args.normalize_fisher_weight,
                                                   minimal_fisher_weight=args.minimal_fisher_weight,
                                                   nums_regmean_examples=args.nums_regmean_examples,
                                                   reduce_non_diagonal_ratio=args.reduce_non_diagonal_ratio,
                                                   param_value_mask_rate=args.param_value_mask_rate,
                                                   weight_format=args.weight_format,
                                                   weight_mask_rates=[args.weight_mask_rate for _ in range(len(models_to_merge))],
                                                   use_weight_rescale=args.use_weight_rescale,
                                                   mask_strategy=args.mask_strategy,
                                                   mask_apply_method=args.mask_apply_method,
                                                   models_use_deepcopy=True)
    merged_model.to('cuda')
    print('======================python====================')
    result,bleu1=do_gen_test(merged_model,tokenizer,task='concode',sub_task='java',type='module_merge')
    print(result)
    bleus.append(bleu1)
    print('======================java====================')
    result,bleu2=do_gen_test(merged_model,tokenizer,task='refine',sub_task='java',type='module_merge')
    print(result)
    bleus.append(bleu2)
    # avg_bleu = write_summary(alphas=[args.scaling_coefficient,args.scaling_coefficient] ,bleus=bleus,output_file=args.log)
    # print(f'Average BLEU: {avg_bleu:.4f}')
    


if __name__ == "__main__":

    args.dataset_names = ['concode','refine' ]
    load_model_paths = [
        # "TransModular_CodeT5/sh/saved_models/summarize/python/codet5_small_all_lr5_bs64_src256_trg128_pat3_e10/checkpoint-best-bleu/pytorch_model.bin",
        # "TransModular_CodeT5/sh/saved_models/concode/codet5_small_all_lr10_bs32_src320_trg150_pat3_e30/checkpoint-best-bleu/pytorch_model.bin"
        
        
        #"TransModular_CodeT5/sh/saved_models/summarize/python/codet5_small_all_lr5_bs64_src256_trg128_pat3_e5/checkpoint-best-bleu/pytorch_model.bin",
        #"TransModular_CodeT5/sh/saved_models/concode/codet5_small_all_lr10_bs32_src320_trg150_pat3_e10/checkpoint-best-bleu/pytorch_model.bin"

        args.model_path1,args.model_path2
    ] 
    models_to_merge = []
    args.language_model_name = 'codet5'
    # put the target dataset name at end
    if args.merging_method_name == "mask_merging":
        args.save_merged_model_path = f"./save_merge_models/{args.dataset_names[0]}_{args.dataset_names[-1]}/{args.merging_method_name}/{args.mask_apply_method}/{args.language_model_name}"
    else:
        args.save_merged_model_path = f"./save_merge_models/{args.dataset_names[0]}_{args.dataset_names[-1]}/{args.merging_method_name}/{args.language_model_name}"
    config_class, model_class, tokenizer_class = MODEL_CLASSES['codet5']
    model_name_or_path = 'TransModular_CodeT5/data/pretrain_model/codet5_small/'
    config = config_class.from_pretrained(model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
    model = model_class.from_pretrained(model_name_or_path)
    merged_model = copy.deepcopy(model)
    for model_path in load_model_paths:
        model_to_merge = copy.deepcopy(model)
        # model_to_merge.to('cuda')
        model_state1 = torch.load(model_path)
        model_to_merge.load_state_dict(model_state1)
        models_to_merge.append(model_to_merge)


    # print(f"models_to_merge: {models_to_merge}")
    merging_method = MergingMethod(merging_method_name=args.merging_method_name)

    # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # put the target dataset name at end
    if args.merging_method_name == "mask_merging":
        save_merge_log_path = f"./save_merge_logs/{args.dataset_names[0]}_{args.dataset_names[-1]}/{args.merging_method_name}/{args.mask_apply_method}/{args.language_model_name}"
    else:
        save_merge_log_path = f"./save_merge_logs/{args.dataset_names[0]}_{args.dataset_names[-1]}/{args.merging_method_name}/{args.language_model_name}"
    os.makedirs(save_merge_log_path, exist_ok=True)
    # create file handler that logs debug and higher level messages
    fh = logging.FileHandler(f"{save_merge_log_path}/{str(time.time())}.log")
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    run_start_time = time.time()
    logger.info(f"********** Run starts. **********")

    best_target_performance = {}
    # search for average_merging
    if args.merging_method_name == "average_merging":
        target_performance = get_merge_performance(args=args, models_to_merge=models_to_merge,  logger=logger, merging_method=merging_method, tokenizer=tokenizer)
        print("***** CUDA.empty_cache() *****")
        torch.cuda.empty_cache()
    # search for task_arithmetic
    elif args.merging_method_name == "task_arithmetic":
        # scaling_coefficient_range = [0.1,0.2, 0.3, 0.5, 0.7, 0.9, 1.0,1.2]
        for alpha1 in [round(x * 0.1, 1) for x in range(5, 14)]:
            for alpha2 in [round(x * 0.1, 1) for x in range(5 , 14)]: 
        # for alpha1 in [0.0,1.0]:
        #     for alpha2 in [0.0,1.0]:    
                args.scaling_coefficients = [alpha1,alpha2]
                # dictionary
                print(f"setup: args.scaling_coefficients : {args.scaling_coefficients}")
                target_performance = get_merge_performance(args=args, models_to_merge=models_to_merge, logger=logger, merging_method=merging_method, tokenizer=tokenizer)
                print("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()
    # search for ties_merging
    elif args.merging_method_name == "ties_merging":
        # scaling_coefficient_range = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]
        param_value_mask_rate_range =[0.5]
        # for scaling_coefficient in scaling_coefficient_range:
        for alpha1 in [round(x * 0.1, 1) for x in range(4, 15)]:
            for alpha2 in [round(x * 0.1, 1) for x in range(4 ,15)]:
        # for alpha1 in [0.0,1.0,2.0,3.0,4.0]:
        #     for alpha2 in [0.0,1.0,2.0,3.0,4.0]:        
                for param_value_mask_rate in param_value_mask_rate_range:
                    args.scaling_coefficients = [alpha1,alpha2]

                    args.param_value_mask_rate = param_value_mask_rate
                    print(f'setup: args.scaling_coefficients :{args.scaling_coefficients},args.param_value_mask_rate:{args.param_value_mask_rate}')
                    # dictionary
                    target_performance = get_merge_performance(args=args, models_to_merge=models_to_merge, logger=logger, merging_method=merging_method, tokenizer=tokenizer)
                    print("***** CUDA.empty_cache() *****")
                    torch.cuda.empty_cache()
    # search for mask_merging
    elif args.merging_method_name == "mask_merging":
        # with open(f"./save_merge_results/{args.dataset_names[0]}_{args.dataset_names[-1]}/{args.mask_apply_method}/{args.language_model_name}.json", "r") as file:
        #     # key is evaluate metric or model hyperparameters
        #     results_dict = json.load(file)
        # if args.mask_apply_method == "task_arithmetic":
        #     args.scaling_coefficient = results_dict["scaling_coefficient"]
        # elif args.mask_apply_method == "fisher_merging":
        #     args.fisher_scaling_coefficients = results_dict["fisher_scaling_coefficients"]
        #     args.nums_fisher_examples = results_dict["nums_fisher_examples"]
        # elif args.mask_apply_method == "regmean_merging":
        #     args.nums_regmean_examples = results_dict["nums_regmean_examples"]
        #     args.reduce_non_diagonal_ratio = results_dict["reduce_non_diagonal_ratio"]
        # elif args.mask_apply_method == "ties_merging":
        #     args.scaling_coefficient = results_dict["scaling_coefficient"]
        #     args.param_value_mask_rate = results_dict["param_value_mask_rate"]

        weight_mask_rate_range = [0.75]
        scaling_coefficient_range = []
        for weight_mask_rate in weight_mask_rate_range:
            args.weight_mask_rate = weight_mask_rate
            # for scaling_coefficient in scaling_coefficient_range:
            for alpha1 in [round(x * 0.1, 1) for x in range(0, 12)]:
                for alpha2 in [round(x * 0.1, 1) for x in range(0 , 12)]:

                    if args.mask_apply_method == "task_arithmetic":
                        # args.scaling_coefficient = scaling_coefficient
                        # for i in range(3):
                        # args.scaling_coefficients = [scaling_coefficient,scaling_coefficient]
                        args.scaling_coefficients = [alpha1,alpha2]
                        print(f"setup: args.scaling_coefficients :{args.scaling_coefficients},weight_mask_rate: {args.weight_mask_rate}")
                        target_performance = get_merge_performance(args=args, models_to_merge=models_to_merge, logger=logger, merging_method=merging_method, tokenizer=tokenizer)
                        print("***** CUDA.empty_cache() *****")
                        torch.cuda.empty_cache()
                # elif args.mask_apply_method == "ties_merging":
                #     args.scaling_coefficient = scaling_coefficient
                #     args.param_value_mask_rate = 1.0
                # dictionary
    else:
        raise NotImplementedError(f"unsupported for merging_method_name {args.merging_method_name}!")
    
    best_target_performance = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in best_target_performance.items()}
    logger.info(f"best performance and configurations on datasets {args.dataset_names}: {best_target_performance}")
    result_json = json.dumps(best_target_performance, indent=4)
    if args.merging_method_name == "mask_merging":
        save_result_dir = f"./save_merge_results/{args.dataset_names[0]}_{args.dataset_names[-1]}/{args.merging_method_name}/{args.mask_apply_method}"
    else:
        save_result_dir = f"./save_merge_results/{args.dataset_names[0]}_{args.dataset_names[-1]}/{args.merging_method_name}"
    os.makedirs(save_result_dir, exist_ok=True)
    save_result_path = os.path.join(save_result_dir, f"{args.language_model_name}.json")
    with open(save_result_path, "w") as file:
        file.write(result_json)

    # avoid the overlap of logs
    logger.removeHandler(fh)
    logger.removeHandler(ch)

    sys.exit()
