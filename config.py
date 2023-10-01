
import configparser

def get_config():
    return dict(
        imgsz= 300,
        n_patches= 36,
        max_seql=50,
        d_model=768,
        nhead=8,
        dim_feedforward=2048,
        num_ts_blocks=4,
        tokenizer_model='medicalai/ClinicalBERT',
        num_epochs=20,
        device='cpu',
        train_batchsz=10,
        val_batchsz=10,
    )


def load_config_ini(config_path):
    config=configparser.ConfigParser()
    config.read(config_path)
    config_=dict()
    for section in [i for i in config if 'DEFAULT' not in i]:
        for param in section:
            if len(config[section][param])>0:
                config_[param]=int(config[section][param]) if config[section][param].isdigit() else config[section][param]
            else:
                config_[param]=None
    return config_




class Config:
    def __init__(self,config_path:str=None):
        self.__dict__=get_config() if not config_path else load_config_ini(config_path)



