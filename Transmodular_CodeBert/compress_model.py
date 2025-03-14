import torch
import sys
import os
from tqdm import tqdm
import numpy as np
import torch.onnx
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


sys.path.append('../../')
from utils import load_model_mlm, load_init_module_sparse,load_init_module_sparse_lr
from global_configs import GlobalConfigs
from transformers import (RobertaModel, RobertaTokenizer)
from torch.utils.data import DataLoader
from datasets import load_from_disk
from mask_layer import MaskLinear, Binarization, init_mask_model
from modeling_roberta_diffhead import DiffheadRobertaModel
from mask_layer import CompressLinear

    
class CompressModel():
    def __init__(self,lang) -> None:
        self.global_configs = GlobalConfigs(lang=lang)
        print(f'global_configs.codebert_base_path: {self.global_configs.codebert_base_path}')
        self.codebert_base = DiffheadRobertaModel.from_pretrained(self.global_configs.codebert_base_path)
        self.tokenizer = RobertaTokenizer.from_pretrained(self.global_configs.codebert_base_path)
        self.tokenizer.do_lower_case = True
        self.module_encoder = load_init_module_sparse_lr(self.codebert_base, f'{self.global_configs.module_path}/pytorch_model.bin')
        self.is_low_rank = True
        self.weight_num = 0 #统计参数量
    
    def process_linear(self,layer, prev_zero=None, layer_flag="linear", search_flag="row", require_pad=False):
        # 输入layer，输入上层layer的全零行索引，用于去除对应列
        # prev_zero是前一个线性层权重W移除的全零行索引
        # flag=“linear”时，使用上一个层的全零行索引去除本层对应列
        # flag=“attention”时，使用上一个层的全零行索引去除本层对应行
        # search_flag是代表搜索零的方向，行还是列，用于前向还是反向搜索
        weights = layer.weight.data.cpu().numpy()
        bias = layer.bias.data.cpu().numpy() if layer.bias is not None else None
        if prev_zero is not None:
            if layer_flag == "linear":
                weights = weights[:,~prev_zero]
            elif layer_flag == "attention":
                weights = weights[~prev_zero]
                bias = bias[~prev_zero]
            else:
                raise ValueError(" layer_flag must be 'linear' or 'attention' ! ")

        if search_flag == "row":
            zero_rows = np.all(weights==0, axis=1)
            # print(f"zero_rows dimention:{len(zero_rows)}")
            non_zero_weights = weights[~zero_rows]
            new_bias = bias[~zero_rows] if bias is not None else None
            next_remove = zero_rows
        elif search_flag == "col":
            zero_cols = np.all(weights==0, axis=0)
            non_zero_weights = weights[:,~zero_cols]
            new_bias = bias[~zero_cols] if bias is not None else None
            next_remove = zero_cols
        else:
            raise ValueError(" search_flag must be 'row' or 'col' ! ")
        # print(f"稀疏度：{np.sum(weights==0)/weights.size}")
        # print(f"全零行：{np.sum(zero_rows)}")
        if require_pad == True:
            new_layer = CompressLinear(non_zero_weights.shape[1], non_zero_weights.shape[0], bias=new_bias is not None)
        else:
            new_layer = nn.Linear(non_zero_weights.shape[1], non_zero_weights.shape[0], bias=new_bias is not None)
        new_layer.weight.data = torch.from_numpy(non_zero_weights)
        new_layer.bias.data = torch.from_numpy(new_bias) if new_bias is not None else None

        return new_layer, next_remove
    

    def plot_weight(self,weight,title,i):
        non_zero_mask = weight != 0
        plt.imshow(non_zero_mask, cmap='Greys', interpolation='none')
        # plt.colorbar()
        plt.title('Weight Matrix')
        plt.savefig(f"./plot_img/Encoder_{i}_{title}.png")

    def count_true_in_segments(self, array, segment_length):
        return [sum(array[i:i + segment_length]) for i in range(0, len(array), segment_length)]

    def compress_encoder(self,is_attention=True):
        self.rank = 16
        model = self.module_encoder
        # encoders = model.encoder.layer
        # 遍历每个encoder
        for i,encoder_layer in enumerate(model.encoder.layer):
            if is_attention:
                ############################################## Attention Layers ##############################################
                # 处理attention中的Q和K,不会改变输出维度
                attention_q_layer = encoder_layer.attention.self.query
                attention_k_layer = encoder_layer.attention.self.key
                # 查找q与k应被去除的索引，不保留新的layer，只合并remove index
                _, next_remove_tmp_q = self.process_linear(attention_q_layer)
                _, next_remove_tmp_k = self.process_linear(attention_k_layer)
                # 求并集并保存 
                union_remove =  np.array([bool(a) or bool(b) for a,b in zip(next_remove_tmp_q,next_remove_tmp_k)])
                print(f'================{len(union_remove)},{len(next_remove_tmp_q)},{len(next_remove_tmp_k)}================')
                # print(self.count_true_in_segments(next_remove, 64))
                new_attention_q_layer,_ = self.process_linear(attention_q_layer,\
                                                            prev_zero=union_remove,\
                                                            layer_flag="attention",\
                                                            require_pad = True)
                new_attention_k_layer,next_remove = self.process_linear(attention_k_layer,\
                                                                        prev_zero=union_remove,\
                                                                        layer_flag="attention",\
                                                                        require_pad = True)
                
                encoder_layer.attention.self.query = new_attention_q_layer
                encoder_layer.attention.self.key = new_attention_k_layer
                encoder_layer.attention.self.query.weight_pad = nn.Parameter(torch.tensor(union_remove, dtype=torch.bool), requires_grad=False)
                encoder_layer.attention.self.key.weight_pad = nn.Parameter(torch.tensor(union_remove, dtype=torch.bool), requires_grad=False)
                # print(encoder_layer.attention.self.key.weight_pad)
                # print(len(encoder_layer.attention.self.key.weight_pad))
                self.weight_num += (new_attention_q_layer.in_features + 1) * new_attention_q_layer.out_features
                self.weight_num += (new_attention_k_layer.in_features + 1) * new_attention_k_layer.out_features

                # 处理attention中的V，会导致attention的输出改变，要跟进下面的维度
                attention_v_layer = encoder_layer.attention.self.value
                new_attention_v_layer, next_remove = self.process_linear(layer=attention_v_layer, require_pad=True)
                encoder_layer.attention.self.value = new_attention_v_layer
                encoder_layer.attention.self.value.weight_pad = nn.Parameter(torch.tensor(next_remove, dtype=torch.bool), requires_grad=False)
                self.weight_num += (new_attention_v_layer.in_features + 1) * new_attention_v_layer.out_features
            else:
                next_remove=None
            ############################################## Output Layers ##############################################
            # 处理 'RobertaSelfOutput' 中的线性层
            self_output_layer = encoder_layer.attention.output.dense
            new_self_output_layer, next_remove_col = self.process_linear(self_output_layer,\
                                                                         prev_zero=next_remove) # 这里要去掉上面v移除的列对应的行
            encoder_layer.attention.output.dense = new_self_output_layer
            prev_zero = next_remove_col
            # self.plot_weight(self_output_layer.weight.data.cpu().numpy(),"RobertaSelfOutput",i)
            self.weight_num += (new_self_output_layer.in_features + 1) * new_self_output_layer.out_features

            # 处理 'RobertaIntermediate' 中的线性层
            intermediate_layer = encoder_layer.intermediate.dense
            new_intermediate_layer,next_remove_col = self.process_linear(intermediate_layer,prev_zero)
            encoder_layer.intermediate.dense = new_intermediate_layer
            prev_zero = next_remove_col
            # self.plot_weight(intermediate_layer.weight.data.cpu().numpy(),"RobertaIntermediate",i)
            self.weight_num += (new_intermediate_layer.in_features + 1) * new_intermediate_layer.out_features

            # 处理 'RobertaOutput' 中的线性层
            output_layer = encoder_layer.output.dense
            new_output_layer,_ = self.process_linear(output_layer,prev_zero)
            encoder_layer.output.dense = new_output_layer
            # self.plot_weight(output_layer.weight.data.cpu().numpy(),"RobertaOutput",i)
            self.weight_num += (new_output_layer.in_features + 1) * new_output_layer.out_features
        
        return model, self.weight_num


# for each_lang in ["go", "java", "javascript", "php", "python", "ruby"]:

each_lang = "python"
print(f'\n\n\n****** LANG = {each_lang} ******\n')
compress_ = CompressModel(each_lang)

if __name__ == '__main__':
    model, weight_num = compress_.compress_encoder()
    torch.save(model.state_dict(), './data_tmp/test.pt')
    # print(model)
    print(f"Encoder权重数为:{weight_num}")
    print(f"总权重数为:{weight_num + 39591168}")
    print(f"原模型Encoder权重数为:{85017600}")
    print(f"原模型权重总数为:{124608768}")
    print(f"encoder权重下降率:{(85017600-weight_num)/85017600}")
