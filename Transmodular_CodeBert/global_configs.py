import os


class GlobalConfigs:
    def __init__(self, lang):
        self.root_dir = f'/home/LAB/longwr/new_SeaM/Tran_SeaM/'  # server4
        if not os.path.exists(self.root_dir):
            self.root_dir = f'/home/binhang/Documents/DNNModularityResearch/TransModular'  # a100-80
            if not os.path.exists(self.root_dir):
                raise ValueError(f'root_dir does not exist!')

        self.data_dir = f'{self.root_dir}/data'
        self.code_search_net_lang_path = f'{self.data_dir}/dataset/code_search_net/{lang}'

        self.pre_trained_model = f'{self.data_dir}/pretrain_model/codebert-base-mlm'
        self.codebert_base_path = f'{self.data_dir}/pretrain_model/codebert-base'
        if not os.path.exists(self.pre_trained_model):
            self.pre_trained_model = 'microsoft/codebert-base-mlm'
        if not os.path.exists(self.codebert_base_path):
            self.codebert_base_path = 'microsoft/codebert-base'

        self.modularizing_config = self.get_modularizing_config()[lang]
        self.module_dir = f'{self.data_dir}/module_{lang}'
        # self.module_path = f"{self.module_dir}/" \
        #                    f"lr_{self.modularizing_config['lr']}_alpha_{self.modularizing_config['alpha']}/result"
        self.module_path = f"/home/LAB/longwr/new_SeaM/Tran_SeaM/data/module_python/lr_0.001_alpha_10.0_ne_4_wrr_24.18/result/"
    def get_modularizing_config(self):
        # "go", "java", "javascript", "php", "python", "ruby"
        modularizing_config = {
            'go': {'lr': 0.01, 'alpha': 10.0, 'n_epochs': 2},
            'java': {'lr': 0.01, 'alpha': 10.0, 'n_epochs': 2},
            'javascript': {'lr': 0.02, 'alpha': 10.0, 'n_epochs': 7},
            'php':  {'lr': 0.01, 'alpha': 5.0, 'n_epochs': 2},
            'python': {'lr': 0.01, 'alpha': 10.0, 'n_epochs': 2},
            'ruby': {'lr': 0.03, 'alpha': 10.0, 'n_epochs': 13}
        }
        return modularizing_config
