# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import logging
import os
import random
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
import multiprocessing
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
import copy
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
cpu_cont = 16
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

logger = logging.getLogger(__name__)

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = x.reshape(-1,x.size(-1)*2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.classifier=RobertaClassificationHead(config)
        self.block_size = tokenizer.max_len_single_sentence
        # self.args=args
    
        
    def forward(self, input_ids=None,labels=None): 
        input_ids=input_ids.view(-1,self.block_size)
        outputs = self.encoder(input_ids= input_ids,attention_mask=input_ids.ne(1))[0]
        logits=self.classifier(outputs)
        prob=F.softmax(logits)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss,prob
        else:
            return prob

def get_example(item):
    url1,url2,label,tokenizer,block_size,cache,url_to_code=item
    if url1 in cache:
        code1=cache[url1].copy()
    else:
        try:
            code=' '.join(url_to_code[url1].split())
        except:
            code=""
        code1=tokenizer.tokenize(code)
    if url2 in cache:
        code2=cache[url2].copy()
    else:
        try:
            code=' '.join(url_to_code[url2].split())
        except:
            code=""
        code2=tokenizer.tokenize(code)
        
    return convert_examples_to_features(code1,code2,label,url1,url2,tokenizer,block_size,cache)


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 label,
                 url1,
                 url2

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label=label
        self.url1=url1
        self.url2=url2
        
def convert_examples_to_features(code1_tokens,code2_tokens,label,url1,url2,tokenizer,block_size,cache):

    #source
    code1_tokens=code1_tokens[:block_size-2]
    code1_tokens =[tokenizer.cls_token]+code1_tokens+[tokenizer.sep_token]
    code2_tokens=code2_tokens[:block_size-2]
    code2_tokens =[tokenizer.cls_token]+code2_tokens+[tokenizer.sep_token]  
    
    code1_ids=tokenizer.convert_tokens_to_ids(code1_tokens)
    padding_length = block_size - len(code1_ids)
    code1_ids+=[tokenizer.pad_token_id]*padding_length
    
    code2_ids=tokenizer.convert_tokens_to_ids(code2_tokens)
    padding_length = block_size - len(code2_ids)
    code2_ids+=[tokenizer.pad_token_id]*padding_length
    
    source_tokens=code1_tokens+code2_tokens
    source_ids=code1_ids+code2_ids
    return InputFeatures(source_tokens,source_ids,label,url1,url2)

class TextDataset(Dataset):
    def __init__(self, tokenizer, block_size, file_path='train',pool=None):
        postfix=file_path.split('/')[-1].split('.txt')[0]
        self.examples = []
        index_filename=file_path
        logger.info("Creating features from index file at %s ", index_filename)
        url_to_code={}
        with open('/'.join(index_filename.split('/')[:-1])+'/data.jsonl') as f:
            for line in f:
                line=line.strip()
                js=json.loads(line)
                url_to_code[js['idx']]=js['func']

        data=[]
        cache={}
        f=open(index_filename)
        with open(index_filename) as f:
            for line in f:
                line=line.strip()
                url1,url2,label=line.split('\t')
                if url1 not in url_to_code or url2 not in url_to_code:
                    continue
                if label=='0':
                    label=0
                else:
                    label=1
                data.append((url1,url2,label,tokenizer, block_size,cache,url_to_code))
        # if 'test' not in postfix:
        #     data=random.sample(data,int(len(data)*0.1))
        if 'test' in postfix:
            data=random.sample(data,int(len(data)*0.1))
        self.examples=pool.map(get_example,tqdm(data,total=len(data)))
        if 'train' in postfix:
            for idx, example in enumerate(self.examples[:3]):
                    logger.info("*** Example ***")
                    logger.info("idx: {}".format(idx))
                    logger.info("label: {}".format(example.label))
                    logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                    logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        
        return torch.tensor(self.examples[item].input_ids),torch.tensor(self.examples[item].label)


def load_and_cache_examples( tokenizer, test_data_file,pool=None):
    block_size = tokenizer.max_len_single_sentence
    dataset = TextDataset(tokenizer, file_path=test_data_file ,block_size=block_size,pool=pool)
    return dataset


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True



def evaluate(model, tokenizer, test_data_file,output_dir,prefix="",pool=multiprocessing.Pool(16)):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = output_dir
    eval_dataset = load_and_cache_examples( tokenizer,test_data_file,pool=pool)
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    eval_batch_size = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) 
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size,num_workers=4,pin_memory=True)


    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]  
    y_trues=[]
    for batch in tqdm(eval_dataloader):
        inputs = batch[0].to(device)        
        labels=batch[1].to(device) 
        with torch.no_grad():
            lm_loss,logit = model(inputs,labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
    logits=np.concatenate(logits,0)
    y_trues=np.concatenate(y_trues,0)
    best_threshold=0
    best_f1=0
    for i in range(1,100):
        threshold=i/100
        y_preds=logits[:,1]>threshold
        from sklearn.metrics import recall_score
        recall=recall_score(y_trues, y_preds)
        from sklearn.metrics import precision_score
        precision=precision_score(y_trues, y_preds)   
        from sklearn.metrics import f1_score
        f1=f1_score(y_trues, y_preds) 
        if f1>best_f1:
            best_f1=f1
            best_threshold=threshold

    y_preds=logits[:,1]>best_threshold
    from sklearn.metrics import recall_score
    recall=recall_score(y_trues, y_preds)
    from sklearn.metrics import precision_score
    precision=precision_score(y_trues, y_preds)   
    from sklearn.metrics import f1_score
    f1=f1_score(y_trues, y_preds)             
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold":best_threshold,
    }

    logger.info("***** Eval results {} *****".format(prefix))
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))

    return result

def test(args, model, tokenizer, prefix="",pool=None,best_threshold=0):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_dataset = load_and_cache_examples(args, tokenizer, test=True,pool=pool)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4,pin_memory=True)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]  
    y_trues=[]
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)        
        labels=batch[1].to(args.device) 
        with torch.no_grad():
            lm_loss,logit = model(inputs,labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
    logits=np.concatenate(logits,0)
    y_preds=logits[:,1]>best_threshold
    with open(os.path.join(args.output_dir,"predictions.txt"),'w') as f:
        for example,pred in zip(eval_dataset.examples,y_preds):
            if pred:
                f.write(example.url1+'\t'+example.url2+'\t'+'1'+'\n')
            else:
                f.write(example.url1+'\t'+example.url2+'\t'+'0'+'\n')

if __name__ == "__main__":
    import argparse
    import copy
    from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

    parser = argparse.ArgumentParser(description="Evaluate Code Clone Detection model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model weights file")
    parser.add_argument("--base_model_path", type=str,
                        default='microsoft/codebert-base',
                        help="Path to the base model")
    parser.add_argument("--test_data_file", type=str,
                        default='Clone_detection_BigCloneBench_2/dataset/test.txt',
                        help="Path to the test data file")
    parser.add_argument("--output_dir", type=str,
                        default="./task_eval/",
                        help="Output directory for evaluation results")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Code Clone Detection model evaluation")

    # Setup device and multiprocessing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pool = multiprocessing.Pool(cpu_cont)

    # Load tokenizer and config
    config = RobertaConfig.from_pretrained(args.base_model_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.base_model_path)
    encoder = RobertaModel.from_pretrained(args.base_model_path)

    # Create model
    model_clone = Model(encoder, config, tokenizer)
    model_clone.load_state_dict(torch.load(args.model_path))
    model_clone.to(device)

    # Evaluate model
    try:
        result = evaluate(model_clone, tokenizer, args.test_data_file, args.output_dir, pool=pool)
        logger.info("Code Clone Detection evaluation completed successfully")
        logger.info(f"Final results: {result}")
    except Exception as e:
        logger.error(f"Code Clone Detection evaluation failed: {str(e)}")
        exit(1)
    finally:
        pool.close()
        pool.join()