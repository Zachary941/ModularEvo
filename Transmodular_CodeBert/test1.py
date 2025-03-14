from global_configs import GlobalConfigs
from transformers import (RobertaModel, RobertaTokenizer)
from modeling_roberta_diffhead import DiffheadRobertaModel
from global_configs import GlobalConfigs
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from datasets import load_from_disk
from utils import load_model_mlm, load_init_module, load_init_module_sparse, load_init_module_sparse_lr
import torch
import time
import sys
sys.path.append('../../')
from compress_model import compress_

global_configs = GlobalConfigs(lang="python")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
codebert_base = DiffheadRobertaModel.from_pretrained(global_configs.codebert_base_path)
tokenizer = RobertaTokenizer.from_pretrained(global_configs.codebert_base_path)
tokenizer.do_lower_case = True

model_encoder = load_model_mlm(codebert_base, global_configs=global_configs)
module_encoder_lr = load_init_module_sparse_lr \
    (codebert_base, "./data/module_python/lr_0.01_alpha_1.5_ne_1_rank_16_lrlm_0.0005/result/pytorch_model.bin")
# module_encoder_compressed = load_init_module_sparse_lr \
#     (codebert_base, "./data_tmp/test.pt")
module_encoder_compressed,_ = compress_.compress_encoder()

lang="python"
batch_size = 16

# for name, param in module_encoder_compressed.named_parameters():
#     print(name)

dataset_path = f'./data/dataset/code_search_net/{lang}_processed'

def Inference_model(model,dataset_path):
    start_time = time.time()
    print(f'Loading the dataset.')
    preprocessed_dataset = load_from_disk(dataset_path)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    data_loader = DataLoader(dataset=preprocessed_dataset['test'], batch_size=batch_size, collate_fn=data_collator)
    model.to(device)
    model.eval()
    print(model)
    print(f'Inference, device:{model.device}')
    for batch in data_loader:
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            sequence_output = outputs.last_hidden_state
            token_output = outputs.last_hidden_state[:, 0, :]
    end_time = time.time()
    return end_time-start_time

model_time = Inference_model(model_encoder,dataset_path)
# module_lr_time = Inference_model(module_encoder_lr,dataset_path)
module_compressed_time = Inference_model(module_encoder_compressed,dataset_path)

print(model_time)
# print(module_lr_time)
print(module_compressed_time)



