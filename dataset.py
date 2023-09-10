import os
import glob
import pandas as pd
import torchvision.io
import torch
from torch.utils.data import Dataset,DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple,Union


def load_tokenizer(tokenizer_model:str):
    if not os.path.exists('data/tokenizer'):
        tokenizer=torch.hub.load('huggingface/pytorch-transformers', 'tokenizer',tokenizer_model)
        tokenizer.save_pretrained('data/tokenizer')
        return tokenizer
    else:
        tokenizer=torch.hub.load('huggingface/pytorch-transformers', 'tokenizer','data/tokenizer/')
        return tokenizer








class Stream_Dataset(Dataset):
    def __init__(self,data_dir:str,img_dir:str='',imgsz:int=300,csv_dir:str=None,tokenizer_model:str='medicalai/ClinicalBERT'):
        self.data_dir=data_dir
        csv_dir=os.path.join(data_dir,csv_dir) if csv_dir else sorted(glob.glob(data_dir+"/*.csv"),reverse=True,key=lambda x:len(pd.read_csv(x)))[0]
        self.data=pd.read_csv(csv_dir).to_numpy()
        self.img_dir=img_dir
        self.imgsz=imgsz
        self.resizer=torchvision.transforms.Resize(size=imgsz)
        self.update_tokenizer(tokenizer_model)
        
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self,idx):
        img_name=self.data[idx,0]
        semantic_str=self.data[idx,1]
        img=self.resizer(torchvision.io.read_image(self.data_dir+'/'+self.img_dir+'/'+img_name))
        return img,semantic_str.strip()
    
    def update_tokenizer(self,tokenizer_model):
        tokenizer=load_tokenizer(tokenizer_model=tokenizer_model)
        _=tokenizer(self.data[:,-1].tolist())
        print(f"Updated tokens from {len(self.data)} rows of data")
        tokenizer.save_pretrained('data/tokenizer/')
        return 

if __name__=='__main__':
    dataset=Stream_Dataset(
            data_dir="./dataset",
            imgsz=300,
            img_dir='images',
            tokenizer_model='bert-base-uncased'
    )
    print(dataset)
    datal=DataLoader(dataset,batch_size=5)
    print(next(iter(datal)))
    print(next(iter(datal))[0].shape)