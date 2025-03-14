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

logger = logging.getLogger(__name__)

cpu_cont = multiprocessing.cpu_count()
pool = multiprocessing.Pool(cpu_cont)

def load_and_cache_gen_data(args, filename, pool, tokenizer, split_tag, only_src=False, is_sample=False):
    # cache the data into args.cache_path except it is sampled
    # only_src: control whether to return only source ids for bleu evaluating (dev/test)
    # return: examples (Example object), data (TensorDataset)
    data_tag = '_all'
    cache_fn = '{}/{}.pt'.format(args.cache_path, split_tag + ('_src' if only_src else '') + data_tag)

    examples = read_examples(filename, args.data_num, args.task)

    if is_sample:
        examples = random.sample(examples, min(5000, len(examples)))
    if split_tag == 'train':
        calc_stats(examples, tokenizer, is_tokenize=True)
    else:
        calc_stats(examples)
    if os.path.exists(cache_fn) and not is_sample:
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        if is_sample:
            logger.info("Sample 5k data for computing bleu from %s", filename)
        else:
            logger.info("Create cache data into %s", cache_fn)
        tuple_examples = [(example, idx, tokenizer, args, split_tag) for idx, example in enumerate(examples)]
        features = pool.map(convert_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        if split_tag == 'test' or only_src:
            data = TensorDataset(all_source_ids)
        else:
            all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
            data = TensorDataset(all_source_ids, all_target_ids)
        if not is_sample:
            torch.save(data, cache_fn)
    return examples, data

def eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, split_tag, criteria):
    logger.info("  ***** Running bleu evaluation on {} data*****".format(split_tag))
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", 64)
    eval_sampler = SequentialSampler(eval_data)

    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=64,
                                     num_workers=4, pin_memory=True)
    model.eval()
    pred_ids = []
    bleu, codebleu = 0.0, 0.0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval bleu for {} set".format(split_tag)):
        source_ids = batch[0].to('cuda')
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        with torch.no_grad():
            if args.model_type == 'roberta':
                preds = model(source_ids=source_ids, source_mask=source_mask)

                top_preds = [pred[0].cpu().numpy() for pred in preds]
            else:
                preds = model.generate(source_ids,
                                       attention_mask=source_mask,
                                       use_cache=True,
                                       num_beams=args.beam_size,
                                       early_stopping=args.task == 'summarize',
                                       max_length=args.max_target_length)
                top_preds = list(preds.cpu().numpy())
            pred_ids.extend(top_preds)

    pred_nls = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]

    output_fn = os.path.join(args.res_dir, "test_{}.output".format(criteria))
    gold_fn = os.path.join(args.res_dir, "test_{}.gold".format(criteria))
    src_fn = os.path.join(args.res_dir, "test_{}.src".format(criteria))

    if args.task in ['defect']:
        target_dict = {0: 'false', 1: 'true'}
        golds = [target_dict[ex.target] for ex in eval_examples]
        eval_acc = np.mean([int(p == g) for p, g in zip(pred_nls, golds)])
        result = {'em': eval_acc * 100, 'bleu': 0, 'codebleu': 0}

        with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
            for pred_nl, gold in zip(pred_nls, eval_examples):
                f.write(pred_nl.strip() + '\n')
                f1.write(target_dict[gold.target] + '\n')
                f2.write(gold.source.strip() + '\n')
            logger.info("Save the predictions into %s", output_fn)
    else:
        dev_accs, predictions = [], []
        with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
            for pred_nl, gold in zip(pred_nls, eval_examples):
                dev_accs.append(pred_nl.strip() == gold.target.strip())
                if args.task in ['summarize']:
                    # for smooth-bleu4 evaluation
                    predictions.append(str(gold.idx) + '\t' + pred_nl)
                    f.write(str(gold.idx) + '\t' + pred_nl.strip() + '\n')
                    f1.write(str(gold.idx) + '\t' + gold.target.strip() + '\n')
                    f2.write(str(gold.idx) + '\t' + gold.source.strip() + '\n')
                else:
                    f.write(pred_nl.strip() + '\n')
                    f1.write(gold.target.strip() + '\n')
                    f2.write(gold.source.strip() + '\n')

        if args.task == 'summarize':
            (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions, gold_fn)
            bleu = round(smooth_bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
        else:
            bleu = round(_bleu(gold_fn, output_fn), 2)
            if args.task in ['concode', 'translate', 'refine']:
                codebleu = calc_code_bleu.get_codebleu(gold_fn, output_fn, args.lang)

        result = {'em': np.mean(dev_accs) * 100, 'bleu': bleu}
        if args.task == 'concode':
            result['codebleu'] = codebleu * 100

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6))

def do_gen_test(model,tokenizer,task,sub_task,type):
    logger.info("  " + "***** Testing *****")
    logger.info("  Batch size = %d", 64)
    if task == 'summarize':
        new_args_dict = {
            'model_name_or_path':'/home/LAB/longwr/new_SeaM/TransModular_CodeT5/data/pretrain_model/codet5_small/',
            'cache_path': f'/home/LAB/longwr/new_SeaM/TransModular_CodeT5/data/{task}/{sub_task}/cache_data',
            'data_num': -1,
            'task': f'{task}',
            'model_type': 'codet5',	
            'add_task_prefix': False,
            'sub_task': 'python',
            'max_source_length': 256,
            'max_target_length': 128,
            'add_lang_ids': False,
            'beam_size':10,
            'res_dir':f'/home/LAB/longwr/new_SeaM/TransModular_CodeT5/task_merge/merge/{type}/{task}/{sub_task}/',
            'test_filename':f'/home/LAB/longwr/new_SeaM/TransModular_CodeT5/data/{task}/{sub_task}/test.jsonl',
        }
    elif task == 'concode':
        new_args_dict = {
            'model_name_or_path':'/home/LAB/longwr/new_SeaM/TransModular_CodeT5/data/pretrain_model/codet5_small/',
            'cache_path': f'/home/LAB/longwr/new_SeaM/TransModular_CodeT5/data/{task}/cache_data',
            'data_num': -1,
            'task': f'{task}',
            'model_type': 'codet5',	
            'add_task_prefix': False,
            'sub_task': 'none',
            'max_source_length': 320,
            'max_target_length': 150,
            'add_lang_ids': False,
            'beam_size':10,
            'res_dir':f'/home/LAB/longwr/new_SeaM/TransModular_CodeT5/task_merge/merge/{type}/{task}/',
            'test_filename':f'/home/LAB/longwr/new_SeaM/TransModular_CodeT5/data/{task}/test.json',
            'lang': 'java',
        }
    elif task == 'refine':
        new_args_dict = {
            'model_name_or_path':'/home/LAB/longwr/new_SeaM/TransModular_CodeT5/data/pretrain_model/codet5_small/',
            'cache_path': f'/home/LAB/longwr/new_SeaM/TransModular_CodeT5/data/{task}/small/cache_data',
            'data_num': -1,
            'task': f'{task}',
            'model_type': 'codet5',	
            'add_task_prefix': False,
            'sub_task': 'none',
            'max_source_length': 130,
            'max_target_length': 120,
            'add_lang_ids': False,
            'beam_size':10,
            'res_dir':f'/home/LAB/longwr/new_SeaM/TransModular_CodeT5/task_merge/merge/{type}/{task}/',
            'test_filename':f'/home/LAB/longwr/new_SeaM/TransModular_CodeT5/data/refine/small/test.buggy-fixed.buggy,/home/LAB/longwr/new_SeaM/TransModular_CodeT5/data/refine/small/test.buggy-fixed.fixed',
            'lang': 'java',
        }
    new_args = argparse.Namespace(**new_args_dict)
    os.makedirs(new_args.res_dir, exist_ok=True)

    for criteria in ['best-bleu']:
        eval_examples, eval_data = load_and_cache_gen_data(new_args, new_args.test_filename, pool, tokenizer, 'test',
                                                            only_src=True, is_sample=False)
        result = eval_bleu_epoch(new_args, eval_data, eval_examples, model, tokenizer, 'test', criteria)
        test_bleu, test_em = result['bleu'], result['em']
        test_codebleu = result['codebleu'] if 'codebleu' in result else 0
        result_str = "[%s] bleu-4: %.2f, em: %.4f, codebleu: %.4f\n" % (criteria, test_bleu, test_em, test_codebleu)
        logger.info(result_str)
    torch.cuda.empty_cache()
    return result_str,test_bleu
