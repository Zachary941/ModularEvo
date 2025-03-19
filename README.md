# ModularEvo: Evolving Multi-Task Models via Neural Network Modularization and Composition

ModularEvo's source code and experimental data.

## Abstract

Deep Neural Network (DNN) models, especially large language models, have been widely applied across various downstream tasks. However, as these models are continuously trained on new data distributions and tasks, they often struggle to effectively integrate new knowledge during their evolution. Traditional full-parameter fine-tuning methods force the model to converge on a single task, resulting in degradation of the original knowledge within the model. Although parameter-efficient fine-tuning methods mitigate this issue by introducing external parameters, they still isolate new knowledge in separate modules, limiting the model’s sustained benefit from downstream tasks.
Inspired by modular design principles in software engineering, we propose ModularEvo, a framework that enables the co-evolution of models and modules. The method first constructs a weight mask for the trained model to accurately identify the parameters related to a specific function, thereby decomposing the model into modules covering different domains and allowing each module to upgrade independently within its domain. However, existing modular approaches often store upgraded modules as independent units, resulting in mutual isolation between the model and its modules. Different from them, ModularEvo not only optimizes each module independently, but also transfers new knowledge acquired by modules in downstream tasks to the model, so as to realize the co-evolution of model modules.
We conducted extensive experiments on various Transformer models covering both classification and generation tasks. The results demonstrate that, compared to baseline methods, ModularEvo achieves an absolute performance increase of 2.34% in multi-round evolution tests, and a 2.48x speedup in downstream task inference, validating the framework’s effectiveness in model evolution and reuse efficiency.

## Requirements
+ advertorch 0.2.3
+ fvcore 0.1.5.post20220512
+ matplotlib 3.4.2
+ numpy 1.19.2
+ python 3.8.10
+ pytorch 1.8.1
+ torchvision 0.9.0
+ tqdm 4.61.0
+ GPU with CUDA support is also needed

## Structure of the directories

```powershell
  |--- README.md                        :  user guidance
  |--- Transmodular_CodeBert/           :  experimental for CodeBert
      |--- modularizer.py				:  modularizer CodeBert
      |--- finetune/					
          |--- finetune_{task}.py		:  finetune on specific tasks
      |--- task_merge/					
          |--- merge_lm.py				:  Knowledge update
  |--- Transmodular_CodeT5/             :  experimental for Codet5
      |--- modularizer.py			    :  modularizer Codet5
      |--- finetune/					
          |--- finetune_{task}.py		:  finetune on specific tasks
      |--- task_merge/					
          |--- merge_lm.py				:  Knowledge update
  |--- Transmodular_GPT-Neo/            :  experimental for GPT-Neo
      |--- modularizer.py				:  modularizer GPT-Neo
      |--- finetune/					
          |--- finetune_{task}.py	    :  finetune on specific tasks
      |--- task_merge/					
          |--- merge_lm.py			    :  Knowledge update
      |--- longrun/
          |--- longrun_finetune.py	    :  finetune on specific tasks
          |--- model_merge.py		    :  Knowledge update
          |--- cost.py		            :  Compute cost

```

The following sections describe how to reproduce the experimental results in our paper. 

## Experimental Workflow

The following sections describe how to reproduce the experimental results in our paper.(Here we take the workflow of GPT-Neo as an example.)

### 1. Model Modularization
 Decompose base model into functional modules .

```bash
# Train specialized modules for different domains by specifying task types:
#   --task [math | law | europarl | github]  
#   math - Mathematics reasoning tasks
#   law - Legal document
#   europarl - Multilingual text processing 
#   github - Source code understanding

# Train independent functional modules
python modularizer.py \
    --task math \  # Domain selection (required)
    --do_train \   # Enable training mode
    --lr 0.005 \   # Learning rate for module specialization
    --n_epochs 4 \ # Epochs per domain adaptation
    --alpha 1 \    # Sparsity regularization weight
    --batch_size 4 # Batchsize for modularization
```

### 2. Downstream Task Fine-tuning

 Finetune modules to specific tasks.

```bash
# Fine-tuning with selective parameter update
# Task domain specifications:
# --task [mathqa | scotus | code | langid]
#   mathqa - Arithmetic problem classfication
#   scotus - Legal classfication
#   code   - Code classfication
#   langid - Language identification 
python finetune_mathqa.py \
    --epoch 2 \         # Fine-tuning iterations
    --lr 5e-5 \         # Learning rate
    --batchsize 8 \     # Batchsize for modularization
    --use_mask \        # Enable mask-guided fine-tuning (only updates masked parameters)
```
### 3.Knowledge update

Knowledge update.

```bash
#modular knowledge
python merge_lm.py --merging_method_name task_arithmetic --language_model_name gptneo --model_path1 module_path1 --model_path2 module_path2 --mask_rate 0.25 --batch_size 8

#baseline
python merge_lm.py --merging_method_name average_merging --language_model_name gptneo --model_path1 model_path1 --model_path2 model_path2 --batch_size 8
python merge_lm.py --merging_method_name task_arithmetic --language_model_name gptneo --model_path1 model_path1 --model_path2 model_path2 --batch_size 8 
python merge_lm.py --merging_method_name ties_merging --language_model_name gptneo --model_path1 model_path1 --model_path2 model_path2 --batch_size 8 --param_value_mask_rate 0.75
python merge_lm.py --merging_method_name mask_merging --language_model_name codet5 --mask_apply_method task_arithmetic --use_weight_rescale --weight_mask_rate 0.75 --model_path1 model_path1 --model_path2 model_path2 --batch_size 8
```



### 4.Multi-iteration Runs

Test method in Multi-iteration Runs

```bash
# first fine-tuning
python longrun_fintune.py --output ./mask_fintune_0 --task_type mathqa  --stage 0 --epochs 2 --tuning_strategy mask
# first knowledge update via composition
python merge_lm.py --merging_method_name task_arithmetic --language_model_name gptneo --model_path1 module_path1 --model_path2 module_path2 --task longrun
# second fine-tuning
python longrun_fintune.py --output ./mask_fintune_0 --task_type mathqa  --stage 1 --epochs 2 --load_model_path first_fintune_module_path --tuning_strategy mask
# second knowledge update via composition
python merge_lm.py --merging_method_name task_arithmetic --language_model_name gptneo --model_path1 module_path1 --model_path2 module_path2 --task longrun
......
```



### 5.Cost Measurement

Assess the inference performance and acceleration capabilities of the modularized components in edge computing scenarios.

```bash
cd ./finetune
python cost.py
```



## Supplementary experimental detail

For reproducibility, we document the λ value(hyperparameter) determined through grid search.

|                        | $\lambda_1$ | $\lambda_2$ |
| ---------------------- | ----------- | ----------- |
| CodeBert(TA)           | 0.5         | 1.1         |
| CodeBert(DARE)         | 0.5         | 1.0         |
| CodeBert(TIES-Merging) | 0.7         | 1.0         |
| CodeBert(Ours)         | 1.0         | 0.7         |
| CodeT5(TA)             | 0.6         | 0.5         |
| CodeT5(DARE)           | 0.6         | 0.5         |
| CodeT5(TIES-Merging)   | 1.1         | 0.9         |
| CodeT5(Ours)           | 0.6         | 0.6         |
| GPT-Neo(TA)            | 0.5         | 0.7         |
| GPT-Neo(DARE)          | 0.6         | 1.0         |
| GPT-Neo(TIES-Merging)  | 0.8         | 1.2         |
| GPT-Neo(Ours)          | 0.7         | 0.9         |

