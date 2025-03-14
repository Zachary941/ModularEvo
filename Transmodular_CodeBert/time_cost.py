from __future__ import absolute_import
import torch
import sys
import os
from tqdm import tqdm

from deepsparse import Engine
from sparseml.pytorch.utils import ModuleExporter
import torch.onnx

sys.path.append('../../')
from utils import load_model_mlm, load_init_module_sparse
from global_configs import GlobalConfigs
from transformers import (RobertaModel, RobertaTokenizer)
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import DataCollatorForLanguageModeling


def main(lang):
    batch_size = 16

    global_configs = GlobalConfigs(lang=lang)
    print(f'global_configs.codebert_base_path: {global_configs.codebert_base_path}')
    codebert_base = RobertaModel.from_pretrained(global_configs.codebert_base_path)
    tokenizer = RobertaTokenizer.from_pretrained(global_configs.codebert_base_path)
    tokenizer.do_lower_case = True

    print(f'Loading the dataset.')
    dataset_path = f'./data/dataset/code_search_net/{lang}_processed'
    preprocessed_dataset = load_from_disk(dataset_path)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    data_loader = DataLoader(dataset=preprocessed_dataset['test'], batch_size=batch_size, collate_fn=data_collator)

    for batch in tqdm(data_loader, ncols=100):
        source_ids, source_mask = batch['input_ids'], batch['attention_mask']
        source_ids, source_mask = source_ids.to(torch.device('cpu')), source_mask.to(torch.device('cpu'))
        break

    intermediate_dir = f'./tmp/{lang}'
    if not os.path.exists(intermediate_dir):
        os.makedirs(intermediate_dir)

    # 1. time cost of the initial model, i.e., the trained model.
    init_model_encoder = load_model_mlm(model=codebert_base, global_configs=global_configs)
    init_model_encoder.to(torch.device('cpu'))
    exporter_init_model_encoder = ModuleExporter(init_model_encoder.eval(), intermediate_dir)
    exporter_init_model_encoder.export_onnx([source_ids, source_mask], name='init_model_encoder.onnx')

    compiled_init_model_encoder = Engine(f'{intermediate_dir}/init_model_encoder.onnx', batch_size=batch_size)
    result = compiled_init_model_encoder.benchmark([source_ids.numpy(), source_mask.numpy()],
                                                   num_iterations=20, num_warmup_iterations=10,
                                                   show_progress=True)
    init_model_ms_per_batch = result.ms_per_batch

    # 2. time cost of the initial module
    # which is decomposed from the initial model (i.e., the trained model) and is not reused on the target task.
    init_module_encoder = load_init_module_sparse(codebert_base, f'{global_configs.module_path}/pytorch_model.bin')
    exporter_init_module_encoder = ModuleExporter(init_module_encoder.eval(), intermediate_dir)
    exporter_init_module_encoder.export_onnx([source_ids, source_mask], name='init_module_encoder.onnx')

    compiled_init_module_encoder = Engine(f'{intermediate_dir}/init_module_encoder.onnx', batch_size=batch_size)
    result = compiled_init_module_encoder.benchmark([source_ids.numpy(), source_mask.numpy()],
                                                    num_iterations=20, num_warmup_iterations=10,
                                                    show_progress=True)
    init_module_ms_per_batch = result.ms_per_batch

    print('='*100)
    print(f'Initial Model: {init_model_ms_per_batch:.2f}ms/batch')
    print(f'Initial Module: {init_module_ms_per_batch:.2f}ms/batch')


# can use `taskset -c ` to run on just the specified cores.
if __name__ == '__main__':
    for each_lang in ["go", "java", "javascript", "php", "python", "ruby"]:
        print(f'\n\n\n****** LANG = {each_lang} ******\n')
        main(each_lang)
