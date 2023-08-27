import torch
import os, logging 

def load_tokenizer(tokenizer_model:str,data_dir:str):
    if not os.path.exists('data/tokenizer'):
        tokenizer=torch.hub.load('huggingface/pytorch-transformers', 'tokenizer',tokenizer_model)
        logging.info("Updating Tokenizer Vocab on Data")
        #insert code to get to process data_dir
        #[tokenizer(i) for i in tqdm(data_dir[:,1])]
        tokenizer.save_pretrained('data/tokenizer')
        return tokenizer
    else:
        tokenizer=torch.hub.load('huggingface/pytorch-transformers', 'tokenizer','data/tokenizer/')
        return tokenizer


def load_bert_embedder(tokenizer_model:str,tokenizer):
    if not os.path.exists('data/model'):
        model=torch.hub.load('huggingface/pytorch-transformers', 'model',tokenizer_model)
        model=update_bert_model(model,tokenizer)
        return model
    else:
        model=torch.hub.load('huggingface/pytorch-transformers', 'model','data/model/')
        model=update_bert_model(model,tokenizer)
        return model




def update_bert_model(model,tokenizer):
    logging.info("Updating Model based on Tokenizer")
    model_vocab_len=model.embeddings.word_embeddings.weight.shape[0]
    if model_vocab_len!=len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    model.save_pretrained('data/model')
    return model







