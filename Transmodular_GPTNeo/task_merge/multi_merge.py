from ast import arg
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
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from merge_utils.merge_utils import set_random_seed
from merge_methods.merging_methods import MergingMethod
from task_eval.law_eval import evaluate_law_model, GPTNeoWithClassificationHead
from task_eval.math_eval import evaluate_math_model
from task_eval.euro_eval import evaluate_euro_model
import time
from itertools import product

timestamp = time.strftime("%Y%m%d_%H%M%S")
# set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser("Interface for merging multiple PLMs")
parser.add_argument("--language_model_name", type=str, default="gpt-neo", help="name of the language model")
parser.add_argument("--merging_method_name", type=str, default="average_merging", help="name of the method to merge models",
                    choices=["average_merging", "task_arithmetic", "fisher_merging", "regmean_merging", "ties_merging", "mask_merging"])
parser.add_argument("--scaling_coefficient", type=float, default=1.0, help="scaling coefficient to merge the task vector")
parser.add_argument("--scaling_coefficients", type=float, nargs="+", help="scaling coefficients for each model being merged")
parser.add_argument("--param_value_mask_rate", type=float, default=0.8, help="mask rate of the smallest-magnitude parameter values")
parser.add_argument("--weight_format", type=str, help="the format of weights to be masked", default="delta_weight", choices=["finetuned_weight", "delta_weight"])
parser.add_argument("--weight_mask_rate", type=float, default=0.1, help="weight mask rate")
parser.add_argument("--use_weight_rescale", action="store_true", default=False, help="whether to rescale the weight by 1 / (1 - weight_mask_rate)")
parser.add_argument("--mask_strategy", type=str, help="mask strategy", default="random", choices=["random", "magnitude"])
parser.add_argument("--mask_apply_method", type=str, default="average_merging", help="merging method that the mask strategy applies",
                    choices=["average_merging", "task_arithmetic", "fisher_merging", "regmean_merging", "ties_merging"])
parser.add_argument("--batch_size", type=int, default=8, help="batch size")
parser.add_argument("--gpu", type=int, default=1, help="number of gpu to use")

parser.add_argument("--model_paths", type=str, nargs="+", help="paths to all models to be merged")
parser.add_argument("--mask_rate", type=float, default=-1, help="rate of mask")
parser.add_argument("--task", type=str, default='', help="longrun or eval")
parser.add_argument("--dataset_names", type=str, nargs="+", help="names of datasets corresponding to each model")
parser.add_argument("--evaluation_tasks", type=str, nargs="+", default=["math", "law"], help="tasks to evaluate the merged model on")

parser.add_argument("--nums_fisher_examples", type=int, nargs="+", help="numbers of examples to compute fisher weights")
parser.add_argument("--fisher_scaling_coefficients", type=float, nargs="+", help="scaling coefficients to merge fisher weights")
parser.add_argument("--normalize_fisher_weight", action="store_true", default=False, help="whether to normalize fisher weights (L2 norm) or not")
parser.add_argument("--minimal_fisher_weight", type=float, default=1e-6, help="the minimal value in fisher weights, used for tackling the potential numerical issues")
parser.add_argument("--nums_regmean_examples", type=int, nargs="+", help="numbers of examples to compute regmean weights")
parser.add_argument("--reduce_non_diagonal_ratio", type=float, default=1.0, help="reduce non-diagonal elements in regmean weights by multiplying this scalar")
parser.add_argument("--num_trials", type=int, default=50, help="number of random trials for large model combinations")

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

def write_summary(alphas, task_results, output_file='summary.txt'):
    """
    Write performance summary to a file
    :param alphas: list of scaling coefficients
    :param task_results: dictionary of task results
    :param output_file: path to output file
    :return: average performance across all tasks
    """
    # Extract task scores
    task_scores = [score for task, score in task_results.items() if isinstance(score, (int, float))]
    
    if not task_scores:
        return 0.0
    
    avg_score = sum(task_scores) / len(task_scores)
    
    if not os.path.exists(output_file):
        with open(output_file, 'w') as f:
            header = []
            header.extend([f"Alpha{i+1}" for i in range(len(alphas))])
            header.extend([f"{task}" for task in task_results.keys() if isinstance(task_results[task], (int, float))])
            header.append("Avg_Score")
            f.write("\t".join(header) + "\n")
            f.write("-" * 80 + "\n")
    
    with open(output_file, 'a') as f:
        row = []
        row.extend([f"{alpha:.4f}" for alpha in alphas])
        row.extend([f"{score:.4f}" for task, score in task_results.items() if isinstance(score, (int, float))])
        row.append(f"{avg_score:.4f}")
        f.write("\t".join(row) + "\n")
    
    return avg_score

def print_params_keys_and_values(params):
    logger.info(f"\n keys and values:")
    for key, value in params.items():
        logger.info(f"{key}")

def add_prefix_to_state_dict_keys(state_dict, prefix):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = f"{prefix}{key}"
        new_state_dict[new_key] = value
    return new_state_dict

def get_merge_performance(args: argparse.Namespace, models_to_merge: list, logger: logging.Logger,
                          merging_method: MergingMethod, tokenizer: transformers.AutoTokenizer):
    """
    get the performance of merging method named merging_method_name
    :param args: ArgumentParser, input argument parser
    :param models_to_merge: list, individual models that need to be merged
    :param logger: Logger, logger
    :param merging_method: MergingMethod, the merging method
    :param tokenizer: AutoTokenizer, tokenizer
    :return: evaluation results
    """
    logger.info(f"configuration is {args}")

    model_name_or_path = "/home/LAB/longwr/new_SeaM/TransModular_GPT/data/gpt-neo-125m/"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    merged_model = GPTNeoWithClassificationHead(model_name_or_path, num_classes=1)
    logger.info("Finish loading model [%s] from %s", get_model_size(merged_model), model_name_or_path)

    # set random seed to guarantee reproducibility
    set_random_seed(seed=42)
    # exclude parameter whose name matches "classification"
    merged_params = merging_method.get_merged_model(merged_model=merged_model,
                                                   models_to_merge=models_to_merge,
                                                   exclude_param_names_regex=[".*classification.*"],
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

    # Evaluate on all tasks specified in evaluation_tasks
    results = {}
    for idx, task in enumerate(args.evaluation_tasks):
        logger.info(f'======================={task}====================')
        if task == "math":
            model_task = GPTNeoWithClassificationHead(model_name_or_path, num_classes=25)
            if idx < len(args.model_paths):
                model_task.load_state_dict(torch.load(args.model_paths[idx]))
            model_task.load_state_dict(merged_params, strict=False)
            result = evaluate_math_model(model_task)
            logger.info(f"{task} _result: {result}")
        elif task == "law":
            model_task = GPTNeoWithClassificationHead(model_name_or_path, num_classes=13)
            if idx < len(args.model_paths):
                model_task.load_state_dict(torch.load(args.model_paths[idx]))
            model_task.load_state_dict(merged_params, strict=False)
            result = evaluate_law_model(model_task)
        elif task == "euro":
            model_task = GPTNeoWithClassificationHead(model_name_or_path, num_classes=6)
            if idx < len(args.model_paths):
                model_task.load_state_dict(torch.load(args.model_paths[idx]))
            model_task.load_state_dict(merged_params, strict=False)
            result = evaluate_euro_model(model_task)
        else:
            logger.warning(f"Unsupported task: {task}, skipping evaluation")
            continue
        
        results[task] = result
    
    # Calculate average performance across all tasks
    task_scores = []
    for task, score in results.items():
        if task == "avg_score":
            continue
        if isinstance(score, dict) and "accuracy" in score:
            task_scores.append(score["accuracy"])

    if task_scores:
        avg_score = sum(task_scores) / len(task_scores)
        results["avg_score"] = avg_score
        logger.info(f"Average accuracy across tasks: {avg_score:.4f}")
        logger.info(f"Individual task accuracies: {task_scores}")
    else:
        logger.warning("No valid accuracy scores found for averaging!")
    
    # Write results to summary file
    if hasattr(args, 'save_merge_log_path'):
        summary_file = os.path.join(args.save_merge_log_path, "performance_summary.txt")
        avg_score = write_summary(args.scaling_coefficients, results, output_file=summary_file)
        logger.info(f"Average performance across tasks: {avg_score:.4f}")
    
    return results

# def generate_coefficient_combinations(num_models, num_trials=None):
#     """
#     Generate coefficient combinations for multiple models
#     :param num_models: number of models to be merged
#     :param num_trials: number of random trials (not used in this implementation)
#     :return: list of coefficient combinations
#     """
#     # Generate coefficients from 0.0 to 1.3 with step 0.1
#     coefficients = [round(x * 0.1, 3) for x in range(0, 13)]  # 0.0 to 1.3
    
#     # For each coefficient value, create a combination where all models have the same coefficient
#     combinations = []
#     for coef in coefficients:
#         combinations.append(tuple([coef] * num_models))
    
    # return combinations
def generate_coefficient_combinations(num_models, num_trials=None):
    """
    Generate coefficient combinations for multiple models where each model can have a different coefficient.
    
    :param num_models: number of models to be merged
    :param num_trials: number of random trials (if specified, will use random sampling)
    :return: list of coefficient combinations
    """
    import itertools
    import random
    
    # Generate coefficient values from 0.3 to 0.8 with step 0.1
    coefficients = [round(0.3 + x * 0.1, 1) for x in range(6)]  # 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
    logger.info(f"Coefficient range: {coefficients}")
    
    # Calculate total possible combinations
    total_combinations = len(coefficients) ** num_models
    logger.info(f"Total possible combinations: {total_combinations}")
    
    logger.info(f"Generating all possible combinations")
    return list(itertools.product(coefficients, repeat=num_models))
    

if __name__ == "__main__":

    # Load models
    models_to_merge = []
    model_name_or_path = "/home/LAB/longwr/new_SeaM/TransModular_GPT/data/gpt-neo-125m/"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load each model with appropriate class based on evaluation tasks
    task_class_mapping = {
        "math": 25,
        "law": 13,
        "euro": 6,
    }
    
    for idx, model_path in enumerate(args.model_paths):
        task = args.evaluation_tasks[idx % len(args.evaluation_tasks)]
        num_classes = task_class_mapping.get(task, 1)
        model = GPTNeoWithClassificationHead(model_name_or_path, num_classes=num_classes)
        model.load_state_dict(torch.load(model_path))
        models_to_merge.append(model)
        logger.info(f'Loaded model from {model_path} for task {task}')
    
    # Set up merging method
    merging_method = MergingMethod(merging_method_name=args.merging_method_name)
    
    # Prepare log directory
    dataset_names_str = "_".join(args.dataset_names)
    if args.mask_rate > 0:
        save_merge_log_path = f"./save_merge_logs_sparity/module_wrr_{args.mask_rate}/{dataset_names_str}_{args.merging_method_name}_{timestamp}"
    else:
        if args.merging_method_name == "mask_merging":
            save_merge_log_path = f"./save_merge_logs_sparity/model/{dataset_names_str}_{args.merging_method_name}_{args.mask_apply_method}_{timestamp}"
        else:
            save_merge_log_path = f"./save_merge_logs_sparity/model/{dataset_names_str}_{args.merging_method_name}_{timestamp}"
    
    args.save_merge_log_path = save_merge_log_path
    os.makedirs(save_merge_log_path, exist_ok=True)
    
    # Set up logging
    fh = logging.FileHandler(f"{save_merge_log_path}/{str(time.time())}.log")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    run_start_time = time.time()
    logger.info(f"********** Run starts for {len(models_to_merge)} models. **********")
    
    best_target_performance = {}
    
    # Handle different merging methods
    if args.merging_method_name == "average_merging":
        # For average merging, all weights are equal
        args.scaling_coefficients = [1.0/len(models_to_merge)] * len(models_to_merge)
        target_performance = get_merge_performance(args=args, models_to_merge=models_to_merge, 
                                                  logger=logger, merging_method=merging_method, 
                                                  tokenizer=tokenizer)
        best_target_performance = target_performance
    
    elif args.merging_method_name == "task_arithmetic":
        # For task arithmetic, try different coefficient combinations
        best_performance = -float('inf')
        coefficient_combinations = generate_coefficient_combinations(len(models_to_merge))
        
        # Create a CSV file to record all performance data
        performance_csv_path = os.path.join(save_merge_log_path, "performance_data.csv")
        with open(performance_csv_path, 'w') as csv_file:
            # Write header
            header = []
            header.extend([f"coefficient_{i+1}" for i in range(len(models_to_merge))])
            header.extend([f"task_{task}" for task in args.evaluation_tasks])
            header.append("avg_score")
            csv_file.write(",".join(header) + "\n")
        
        logger.info(f"Evaluating {len(coefficient_combinations)} coefficient combinations")
        
        all_performances = []
        
        for coeffs in coefficient_combinations:
            args.scaling_coefficients = list(coeffs)
            logger.info(f"Testing coefficients: {args.scaling_coefficients}")
            
            target_performance = get_merge_performance(args=args, models_to_merge=models_to_merge,
                                                     logger=logger, merging_method=merging_method,
                                                     tokenizer=tokenizer)
            
            # Record performance in CSV
            with open(performance_csv_path, 'a') as csv_file:
                row = []
                row.extend([f"{coef:.4f}" for coef in args.scaling_coefficients])
                row.extend([f"{target_performance.get(task, 'N/A')}" for task in args.evaluation_tasks])
                avg_score = target_performance.get("avg_score", 0.0)
                row.append(f"{avg_score:.4f}")
                csv_file.write(",".join(row) + "\n")
            
            # Store performance data for later analysis
            perf_entry = {
                "coefficients": args.scaling_coefficients.copy(),
                "performance": target_performance.copy()
            }
            all_performances.append(perf_entry)
            
            # Update best performance if better
            if "avg_score" in target_performance and target_performance["avg_score"] > best_performance:
                best_performance = target_performance["avg_score"]
                best_target_performance = target_performance
                best_target_performance["scaling_coefficients"] = args.scaling_coefficients.copy()
            
            # Clean up memory
            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()
        
        # Also evaluate special cases: each model individually
        for i in range(len(models_to_merge)):
            coeffs = [0.0] * len(models_to_merge)
            coeffs[i] = 1.0
            args.scaling_coefficients = coeffs
            logger.info(f"Testing individual model {i}: {args.scaling_coefficients}")
            
            target_performance = get_merge_performance(args=args, models_to_merge=models_to_merge,
                                                     logger=logger, merging_method=merging_method,
                                                     tokenizer=tokenizer)
            
            # Record performance in CSV
            with open(performance_csv_path, 'a') as csv_file:
                row = []
                row.extend([f"{coef:.4f}" for coef in args.scaling_coefficients])
                row.extend([f"{target_performance.get(task, 'N/A')}" for task in args.evaluation_tasks])
                avg_score = target_performance.get("avg_score", 0.0)
                row.append(f"{avg_score:.4f}")
                csv_file.write(",".join(row) + "\n")
            
            # Store performance data
            perf_entry = {
                "coefficients": args.scaling_coefficients.copy(),
                "performance": target_performance.copy()
            }
            all_performances.append(perf_entry)
            
            # Update best performance if better
            if "avg_score" in target_performance and target_performance["avg_score"] > best_performance:
                best_performance = target_performance["avg_score"]
                best_target_performance = target_performance
                best_target_performance["scaling_coefficients"] = args.scaling_coefficients.copy()
            
            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()
        
        # Save all performance data as JSON for easier analysis
        all_performances_json_path = os.path.join(save_merge_log_path, "all_performances.json")
        with open(all_performances_json_path, 'w') as json_file:
            json.dump(all_performances, json_file, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
        
        logger.info(f"Best average performance: {best_performance:.4f} with coefficients: {best_target_performance.get('scaling_coefficients', [])}")
    
    elif args.merging_method_name == "ties_merging":
        # Similar modifications for ties_merging...

        # Create a CSV file to record all performance data
        performance_csv_path = os.path.join(save_merge_log_path, "performance_data.csv")
        with open(performance_csv_path, 'w') as csv_file:
            # Write header
            header = []
            header.extend([f"coefficient_{i+1}" for i in range(len(models_to_merge))])
            header.append("param_value_mask_rate")
            header.extend([f"task_{task}" for task in args.evaluation_tasks])
            header.append("avg_score")
            csv_file.write(",".join(header) + "\n")
        
        param_value_mask_rate_range = [0.75]
        best_performance = -float('inf')
        all_performances = []
        
        for param_value_mask_rate in param_value_mask_rate_range:
            args.param_value_mask_rate = param_value_mask_rate
            
            coefficient_combinations = generate_coefficient_combinations(len(models_to_merge))
            
            for coeffs in coefficient_combinations:
                args.scaling_coefficients = list(coeffs)
                logger.info(f"Testing coefficients: {args.scaling_coefficients}, param_value_mask_rate: {args.param_value_mask_rate}")
                
                target_performance = get_merge_performance(args=args, models_to_merge=models_to_merge,
                                                         logger=logger, merging_method=merging_method,
                                                         tokenizer=tokenizer)
                
                # Record performance in CSV
                with open(performance_csv_path, 'a') as csv_file:
                    row = []
                    row.extend([f"{coef:.4f}" for coef in args.scaling_coefficients])
                    row.append(f"{param_value_mask_rate:.4f}")
                    row.extend([f"{target_performance.get(task, 'N/A')}" for task in args.evaluation_tasks])
                    avg_score = target_performance.get("avg_score", 0.0)
                    row.append(f"{avg_score:.4f}")
                    csv_file.write(",".join(row) + "\n")
                
                # Store performance data
                perf_entry = {
                    "coefficients": args.scaling_coefficients.copy(),
                    "param_value_mask_rate": param_value_mask_rate,
                    "performance": target_performance.copy()
                }
                all_performances.append(perf_entry)
                
                # Update best performance if better
                if "avg_score" in target_performance and target_performance["avg_score"] > best_performance:
                    best_performance = target_performance["avg_score"]
                    best_target_performance = target_performance
                    best_target_performance["scaling_coefficients"] = args.scaling_coefficients.copy()
                    best_target_performance["param_value_mask_rate"] = args.param_value_mask_rate
                
                # Clean up memory
                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()
        
        # Save all performance data as JSON for easier analysis
        all_performances_json_path = os.path.join(save_merge_log_path, "all_performances.json")
        with open(all_performances_json_path, 'w') as json_file:
            json.dump(all_performances, json_file, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
    
    elif args.merging_method_name == "mask_merging":
        # Create a CSV file to record all performance data
        performance_csv_path = os.path.join(save_merge_log_path, "performance_data.csv")
        with open(performance_csv_path, 'w') as csv_file:
            # Write header
            header = []
            header.extend([f"coefficient_{i+1}" for i in range(len(models_to_merge))])
            header.append("weight_mask_rate")
            header.extend([f"task_{task}" for task in args.evaluation_tasks])
            header.append("avg_score")
            csv_file.write(",".join(header) + "\n")
            
        weight_mask_rate_range = [0.75]
        best_performance = -float('inf')
        all_performances = []
        
        for weight_mask_rate in weight_mask_rate_range:
            args.weight_mask_rate = weight_mask_rate
            
            coefficient_combinations = generate_coefficient_combinations(len(models_to_merge))
            
            for coeffs in coefficient_combinations:
                if args.mask_apply_method == "task_arithmetic":
                    args.scaling_coefficients = list(coeffs)
                    logger.info(f"Testing coefficients: {args.scaling_coefficients}, weight_mask_rate: {args.weight_mask_rate}")
                    
                    target_performance = get_merge_performance(args=args, models_to_merge=models_to_merge,
                                                             logger=logger, merging_method=merging_method,
                                                             tokenizer=tokenizer)
                    
                    # Record performance in CSV
                    with open(performance_csv_path, 'a') as csv_file:
                        row = []
                        row.extend([f"{coef:.4f}" for coef in args.scaling_coefficients])
                        row.append(f"{weight_mask_rate:.4f}")
                        row.extend([f"{target_performance.get(task, 'N/A')}" for task in args.evaluation_tasks])
                        avg_score = target_performance.get("avg_score", 0.0)
                        row.append(f"{avg_score:.4f}")
                        csv_file.write(",".join(row) + "\n")
                    
                    # Store performance data
                    perf_entry = {
                        "coefficients": args.scaling_coefficients.copy(),
                        "weight_mask_rate": weight_mask_rate,
                        "performance": target_performance.copy()
                    }
                    all_performances.append(perf_entry)
                    
                    # Update best performance if better
                    if "avg_score" in target_performance and target_performance["avg_score"] > best_performance:
                        best_performance = target_performance["avg_score"]
                        best_target_performance = target_performance
                        best_target_performance["scaling_coefficients"] = args.scaling_coefficients.copy()
                        best_target_performance["weight_mask_rate"] = args.weight_mask_rate
                    
                    # Clean up memory
                    logger.info("***** CUDA.empty_cache() *****")
                    torch.cuda.empty_cache()
        
        # Save all performance data as JSON for easier analysis
        all_performances_json_path = os.path.join(save_merge_log_path, "all_performances.json")
        with open(all_performances_json_path, 'w') as json_file:
            json.dump(all_performances, json_file, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)

    best_target_performance = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in best_target_performance.items()}
    logger.info(f"Best performance and configurations on datasets {args.dataset_names}: {best_target_performance}")
    
    # Save results
    result_json = json.dumps(best_target_performance, indent=4)
    save_result_dir = save_merge_log_path
    os.makedirs(save_result_dir, exist_ok=True)
    save_result_path = os.path.join(save_result_dir, f"{args.language_model_name}.json")
    with open(save_result_path, "w") as file:
        file.write(result_json)
    
    # Clean up logging
    logger.removeHandler(fh)
    logger.removeHandler(ch)
    
    sys.exit()
