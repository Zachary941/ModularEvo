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
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import json
import logging
import os
import sys
from io import open
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, TensorDataset
import torch.nn as nn
from torch.autograd import Variable
import copy

import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.modeling_utils import PreTrainedModel
csv.field_size_limit(sys.maxsize)
logger = logging.getLogger(__name__)

class Model(PreTrainedModel):
    def __init__(self, encoder, config, tokenizer):
        super(Model, self).__init__(config)
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.mlp = nn.Sequential(nn.Linear(768*4, 768),
                                 nn.Tanh(),
                                 nn.Linear(768, 1),
                                 nn.Sigmoid())
        self.loss_func = nn.BCELoss()
        # self.args = args

    def forward(self, code_inputs, nl_inputs, labels, return_vec=False):
        bs = code_inputs.shape[0]
        inputs = torch.cat((code_inputs, nl_inputs), 0)
        outputs = self.encoder(inputs, attention_mask=inputs.ne(1))[1]
        code_vec = outputs[:bs]
        nl_vec = outputs[bs:]
        if return_vec:
            return code_vec, nl_vec

        logits = self.mlp(torch.cat((nl_vec, code_vec, nl_vec-code_vec, nl_vec*code_vec), 1))
        logits = logits.view(-1)
        loss = self.loss_func(logits, labels.float())
        predictions = (logits > 0.5).int()  # (Batch, )
        return loss,predictions




class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self, code_tokens, code_ids, nl_tokens, nl_ids, label, idx):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.label = label
        self.idx = idx


class InputFeaturesTriplet(InputFeatures):
    """A single training/test features for a example. Add docstring seperately. """
    def __init__(self, code_tokens, code_ids, nl_tokens, nl_ids, ds_tokens, ds_ids, label, idx):
        super(InputFeaturesTriplet, self).__init__(code_tokens, code_ids, nl_tokens, nl_ids, label, idx)
        self.ds_tokens = ds_tokens
        self.ds_ids = ds_ids


def convert_examples_to_features(js, tokenizer,max_seq_length):

    # label
    label = js['label']

    # code
    code = js['code']
    code_tokens = tokenizer.tokenize(code)[:max_seq_length-2]
    code_tokens = [tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = max_seq_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length

    nl = js['doc']  # query
    nl_tokens = tokenizer.tokenize(nl)[:max_seq_length-2]
    nl_tokens = [tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = max_seq_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id]*padding_length

    return InputFeatures(code_tokens, code_ids, nl_tokens, nl_ids, label, js['idx'])

class TextDataset(Dataset):
    def __init__(self, tokenizer,max_seq_length,file_path=None, type=None):
        # json file: dict: idx, query, doc, code
        self.examples = []
        self.type = type
        data=[]
        with open(file_path, 'r') as f:
            data = json.load(f)
        if self.type == 'test':
            for js in data:
                js['label'] = 0
        for js in data:
            self.examples.append(convert_examples_to_features(js, tokenizer,max_seq_length))
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120','_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                logger.info("nl_tokens: {}".format([x.replace('\u0120','_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        """ return both tokenized code ids and nl ids and label"""
        return torch.tensor(self.examples[i].code_ids), \
               torch.tensor(self.examples[i].nl_ids),\
               torch.tensor(self.examples[i].label)




def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    prec = precision_score(y_true=labels, y_pred=preds)
    reca = recall_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "precision": prec,
        "recall": reca,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "webquery":
        return acc_and_f1(preds, labels)
    if task_name == "staqc":
        return acc_and_f1(preds, labels)
    else:
        raise KeyError(task_name)


def evaluate( model, tokenizer,eval_data_file=None,output_dir=None):
    eval_output_dir = output_dir
    eval_data_path = os.path.join(eval_data_file)
    max_seq_length = tokenizer.max_len_single_sentence
    eval_dataset = TextDataset(tokenizer,max_seq_length,eval_data_path, type='eval')
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    eval_batch_size = 8 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) 
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size, num_workers=4, pin_memory=True)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    all_predictions = []
    all_labels = []
    for batch in eval_dataloader:
        code_inputs = batch[0].to(device)
        nl_inputs = batch[1].to(device)
        labels = batch[2].to(device)
        with torch.no_grad():
            lm_loss, predictions = model(code_inputs, nl_inputs, labels)
            # lm_loss,code_vec,nl_vec = model(code_inputs,nl_inputs)
            eval_loss += lm_loss.mean().item()
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
        nb_eval_steps += 1
    all_predictions = torch.cat(all_predictions, 0).squeeze().numpy()
    all_labels = torch.cat(all_labels, 0).squeeze().numpy()
    eval_loss = torch.tensor(eval_loss / nb_eval_steps)

    results = acc_and_f1(all_predictions, all_labels)
    results.update({"eval_loss": float(eval_loss)})
    return results