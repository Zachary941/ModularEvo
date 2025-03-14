#%%
"""This project aims to modularize GPT-Neo."""

from datasets import load_dataset, load_from_disk
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from mask_layer import init_mask_model


# 预处理pile的github数据集
def preprocess_pile(raw_dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding=True)

    pile_data = raw_dataset.map(tokenize_function, batched=True, remove_columns=raw_dataset['train'].column_names,
                                desc='preprocess', num_proc=8)

    pile_data['train'] = pile_data['train'].add_column('labels', pile_data['train']['input_ids'])
    pile_data['test'] = pile_data['test'].add_column('labels', pile_data['test']['input_ids'])
    return pile_data


def load_data(lang):
    # 数据集路径，是单个语言的jsonl路径
    # 需要分开加载train和test，load_dataset会打train和test标签
    if lang == 'python':
        dataset_path_gpt_train = f'data/dataset/pile/train/Github_train/python_train.jsonl'
        dataset_path_gpt_test = f'data/dataset/pile/test/Github_test/python_test.jsonl'
    elif lang == 'law':
        dataset_path_gpt_train = f'data/dataset/pile/train/FreeLaw_train.jsonl'
        dataset_path_gpt_test = f'data/dataset/pile/test/FreeLaw_test.jsonl'
    else:
        raise ValueError
    data_file_gpt = {"train": dataset_path_gpt_train, "test": dataset_path_gpt_test}

    pile_data = load_dataset('json', data_files=data_file_gpt)

    # 把数据加载到列表，后面要分割train和test数据集
    print("+++++++++++++++ Raw +++++++++++++++")
    print(f"pile_data type:{type(pile_data)}")
    print(pile_data["test"][0])
    print("===================================")
    print(pile_data)
    return pile_data


model_path_gpt = f"./data/gpt-neo-125m"
model_gpt = GPTNeoForCausalLM.from_pretrained(model_path_gpt)
# module = init_mask_model(model=model_gpt, no_mask=['lm_head'])  # TEST
module = init_mask_model(model=model_gpt, no_mask=[])
print(f'\n\n{module}\n\n')

tokenizer_gpt = GPT2Tokenizer.from_pretrained(model_path_gpt)

tokenizer_gpt.pad_token = tokenizer_gpt.eos_token

pile_data = load_data(lang='python')
processed_lang_data = preprocess_pile(pile_data, tokenizer_gpt)
print(processed_lang_data)

#%%
import copy

def clip_processed_data(processed_data, n_train, n_test):
    part_processed_data = copy.deepcopy(processed_data)
    if n_train > 0:
        part_train_data = processed_data['train'].select(list(range(n_train)))
        part_processed_data['train'] = part_train_data

    if n_test > 0:
        part_test_data = processed_data['test'].select(list(range(n_test)))
        part_processed_data['test'] = part_test_data
    print(part_processed_data)
    return part_processed_data

# part_train_data = processed_lang_data['train'].select([0, 1, 2, 3, 4])
# print(part_train_data)
#
# part_test_data = processed_lang_data['test'].select([0, 1, 2, 3, 4])
# print(part_test_data)
#
# part_processed_lang_data = copy.deepcopy(processed_lang_data)
# print(part_processed_lang_data)
#
# part_processed_lang_data['train'] = part_train_data
# part_processed_lang_data['test'] = part_test_data
part_processed_lang_data = clip_processed_data(processed_lang_data, n_train=100, n_test=100)
print(part_processed_lang_data)