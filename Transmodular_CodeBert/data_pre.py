import argparse
import os
from transformers import RobertaTokenizer, RobertaForMaskedLM, pipeline
from datasets import load_dataset, load_from_disk


dataset_path='/home/LAB/longwr/new_SeaM/Tran_SeaM/data/dataset/code_search_net_2/data/'
code_search_net = load_dataset(dataset_path, 'java')
code_search_net.save_to_disk(f'{dataset_path}/java2')
# code_search_net = load_from_disk(dataset_path)