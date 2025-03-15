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
  |--- Transmodular_CodeBert/           :  experimental data
  |--- Transmodular_CodeT5/           :  experimental data
  |--- Transmodular_GPT-Neo/           :  experimental data

```

The following sections describe how to reproduce the experimental results in our paper. 

## Experimental Workflow

Our framework consists of three core stages with additional reproducibility scripts:

### 1. Model Modularization
**Objective**: Decompose base model into functional modules  

```bash
# Train independent modules
python modularizer_t5.py  --lang task --do_train --n_epochs n_epochs --alpha 1 --lr lr
```

### 2. Downstream Task Fine-tuning

**Objective**: Adapt modules to specific tasks

```bash
# Fine-tuning
# full fine-tuning
python run_exp.py --model_tag codet5_small --task clone --sub_task none
# modular fine-tuning
python run_exp_module.py --model_tag codet5_small --task clone --sub_task none
```
### 3.Knowledge update

**Objective**: Knowledge update

```bash
python merge_lm.py --merging_method_name task_arithmetic --language_model_name codet5 
```



### 4.Multi-iteration Runs

**Objective**: test method in Multi-iteration Runs

```bash
# full fine-tuning
python longrun.py --tuning_strategy full --merge_method task_arithmetic --alpha1 0.5 --alpha2 0.5
# modular fine-tuning
python longrun.py --tuning_strategy mask --merge_method task_arithmetic --alpha1 0.5 --alpha2 0.5
```



### 5.Cost Measurement

**Objective**: 

```bash
python cost.py
```



