import torch
import os, logging 
from torch.optim import Adam,SGD,RMSprop
from config import Config
config=Config('config.ini')

# def load_tokenizer(tokenizer_model:str,data_dir:str):
#     if not os.path.exists('data/tokenizer'):
#         tokenizer=torch.hub.load('huggingface/pytorch-transformers', 'tokenizer',tokenizer_model)
#         logging.info("Updating Tokenizer Vocab on Data")
#         #insert code to get to process data_dir
#         #[tokenizer(i) for i in tqdm(data_dir[:,1])]
#         tokenizer.save_pretrained('data/tokenizer')
#         return tokenizer
#     else:
#         tokenizer=torch.hub.load('huggingface/pytorch-transformers', 'tokenizer','data/tokenizer/')
#         return tokenizer

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask




def load_bert_embedder(tokenizer_model:str,tokenizer=None):
    tokenizer_file_name= tokenizer.name if tokenizer else tokenizer_model
    assert tokenizer_file_name==tokenizer_model, "Please make sure the HuggingFace tokenizer and embedder are from same repo"
    mod_pth=os.path.join("data",tokenizer_file_name.split('/')[-1],"model")
    model=torch.hub.load('huggingface/pytorch-transformers', 'model',(mod_pth if os.path.exists(mod_pth) else tokenizer_model))
    # model=torch.hub.load('huggingface/pytorch-transformers', 'model',mod_pth) 
    model=update_bert_model(model,tokenizer,tokenizer_model)
    return model



def update_bert_model(model,tokenizer=None,tokenizer_model=None):
    logging.info("Updating Model based on Tokenizer")
    tokenizer_file_name=tokenizer.name if not tokenizer_model else tokenizer_model
    tok_pth=os.path.join("data",tokenizer_file_name.split('/')[-1],"tokenizer")
    if not tokenizer:
        if not os.path.exists(tok_pth):
            raise ValueError(f"Please update tokenizer with desired dataset and keep at {tok_pth}")
        else:
            tokenizer=torch.hub.load('huggingface/pytorch-transformers', 'model',tok_pth)
    model_vocab_len=model.embeddings.word_embeddings.weight.shape[0]
    if model_vocab_len<len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    model.save_pretrained(tok_pth.replace('tokenizer','model'))
    return model




def load_optimizer(opt):
    if opt.lower()=='adam':
        return Adam(lr=config.learning_rate)
    elif opt.lower()=='sgd':
        return SGD(lr=config.learning_rate,momentum=config.sgd_momentum)
    elif opt.lower()=='rmsprop':
        return RMSprop(lr=config.learning_rate,momentum=config.momentum,alpha=config.rms_alpha)


