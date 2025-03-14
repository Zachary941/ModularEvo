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

from sklearn.metrics import recall_score, precision_score, f1_score
sys.path.append('../../')
sys.path.append('../')
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
                 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
                 'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer)}
from evaluator import smooth_bleu
from evaluator.CodeBLEU import calc_code_bleu
from evaluator.bleu import _bleu
from utils import read_examples, convert_examples_to_features, calc_stats,convert_clone_examples_to_features
from configs import add_args, set_seed, set_dist

logger = logging.getLogger(__name__)
cpu_cont = multiprocessing.cpu_count()
pool = multiprocessing.Pool(cpu_cont)

def load_and_cache_clone_data(args, filename, pool, tokenizer, split_tag, is_sample=False):
    cache_fn = '{}/{}.pt'.format(args.cache_path, split_tag + '_all' if args.data_num == -1 else '_%d' % args.data_num)
    examples = read_examples(filename, args.data_num, args.task)
    if is_sample:
        examples = random.sample(examples, int(len(examples) * 0.1))

    calc_stats(examples, tokenizer, is_tokenize=True)
    if os.path.exists(cache_fn):
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        if is_sample:
            logger.info("Sample 10 percent of data from %s", filename)
        elif args.data_num == -1:
            logger.info("Create cache data into %s", cache_fn)
        tuple_examples = [(example, idx, tokenizer, args) for idx, example in enumerate(examples)]
        features = pool.map(convert_clone_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        data = TensorDataset(all_source_ids, all_labels)

        if args.local_rank in [-1, 0] and args.data_num == -1:
            torch.save(data, cache_fn)
    return examples, data

def evaluate(args, model, eval_examples, eval_data, write_to_pred=False):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation  *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Evaluating"):
        inputs = batch[0].to(args.device)
        labels = batch[1].to(args.device)
        with torch.no_grad():
            lm_loss, logit = model(inputs, labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    best_threshold = 0.5

    y_preds = logits[:, 1] > best_threshold
    recall = recall_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds)
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold": best_threshold,
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))
    logger.info("  " + "*" * 20)

    if write_to_pred:
        with open(os.path.join(args.output_dir, "predictions.txt"), 'w') as f:
            for example, pred in zip(eval_examples, y_preds):
                if pred:
                    f.write(example.url1 + '\t' + example.url2 + '\t' + '1' + '\n')
                else:
                    f.write(example.url1 + '\t' + example.url2 + '\t' + '0' + '\n')

    return result

def do_clone_test(model,tokenizer,task,sub_task,type):
    new_args_dict = {
        'model_name_or_path':'TransModular_CodeT5/data/pretrain_model/codet5_small/',
        'cache_path': f'TransModular_CodeT5/data/{task}/cache_data',
        'data_num': -1,
        'task': f'{task}',
        'model_type': 'codet5',	
        'add_task_prefix': False,
        'max_source_length': 400,
        'max_target_length': 400,
        'add_lang_ids': False,
        'beam_size':10,
        'res_dir':f'TransModular_CodeT5/task_merge/merge/{type}/{task}',
        'test_filename':f'TransModular_CodeT5/data/{task}/test.txt',
        'eval_batch_size':25
    }
    new_args = argparse.Namespace(**new_args_dict)
    logger.info("  " + "***** Testing *****")
    logger.info("  Batch size = %d", new_args.eval_batch_size)

    for criteria in ['best-f1']:
        eval_examples, eval_data = load_and_cache_clone_data(new_args, new_args.test_filename, pool, tokenizer, 'test',
                                                                False)

        result = evaluate(new_args, model, eval_examples, eval_data, write_to_pred=True)
        logger.info("  test_f1=%.4f", result['eval_f1'])
        logger.info("  test_prec=%.4f", result['eval_precision'])
        logger.info("  test_rec=%.4f", result['eval_recall'])
        logger.info("  " + "*" * 20)
    return result