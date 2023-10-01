from torch import nn
import torch
from torchmetrics.text import BLEUScore, ROUGEScore,WordErrorRate
from dataset import Stream_Dataset,HFTokenizer
from config import load_config_ini, get_config, Config

config_path='config.ini'
config=Config(config_path=config_path)
tokenizer=HFTokenizer(tokenizer_model=config.tokenizer_model)
rouge_l=ROUGEScore(rouge_keys=['rougeL'])
bleu4=BLEUScore(n_grams=4)
wer=WordErrorRate
