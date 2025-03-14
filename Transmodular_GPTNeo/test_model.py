from transformers import GPT2Tokenizer
from transformers import RobertaTokenizer

model_path_GPT = f"./data/gpt-neo-125m"
model_path = f'./data/codebert-base-mlm'
tokenizer_GPT = GPT2Tokenizer.from_pretrained(model_path_GPT)
tokenizer = RobertaTokenizer.from_pretrained(model_path)
print("RobertaTokenizer tokenizer max length:",tokenizer.model_max_length)
print("GPT2Tokenizer tokenizer max length:",tokenizer_GPT.model_max_length)
