import os
import glob
import pandas as pd
import torchvision.io
import torch
from torch.utils.data import Dataset,DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple,Union,Optional
from torchvision import transforms
import numpy as np
from config import get_config

def load_tokenizer(tokenizer_model:str):
    if not os.path.exists('data/tokenizer'):
        tokenizer=torch.hub.load('huggingface/pytorch-transformers', 'tokenizer',tokenizer_model)
        tokenizer.save_pretrained('data/tokenizer')
        return tokenizer
    else:
        tokenizer=torch.hub.load('huggingface/pytorch-transformers', 'tokenizer','data/tokenizer/')
        return tokenizer


class HFTokenizer:
    def __init__(self,tokenizer_model,max_length):
        tok_pth=os.path.join("data",tokenizer_model.split('/')[-1],"tokenizer")
        self.name=tokenizer_model
        self.tokenizer=torch.hub.load('huggingface/pytorch-transformers', 'tokenizer',tokenizer_model) if not os.path.exists(tok_pth) else torch.hub.load('huggingface/pytorch-transformers', 'tokenizer',tok_pth)
        self.save_pth=tok_pth
        if not max_length:
            max_length=(get_config())['max_seql']
        self.tokenizer.model_max_length=max_length
        self.vocab=self.tokenizer.vocab
    def __call__(self,data,padding=None,truncation=True,return_tensors='pt',add_special_tokens=True):
        return self.tokenizer(data,padding=padding,truncation=truncation,return_tensors=return_tensors,add_special_tokens=add_special_tokens).input_ids
    
    def decode(self,tok_ids):
        return self.tokenizer.decode(tok_ids,skip_special_tokens=True,clean_up_tokenization_spaces=True)
    
    def __len__(self):
        return len(self.tokenizer)

    def save(self):
        self.tokenizer.save_pretrained(self.save_pth)
    
    def update(self,data):
        dpoints=0
        if isinstance(data,torch.utils.data.dataloader.DataLoader):
            for (img,text) in data:
                self.tokenizer(text)
                dpoints+=len(text)
        elif isinstance(data,(np.ndarray,torch.Tensor)):
            data=data.tolist()
            self.tokenizer(data)
            dpoints+=len(data)
        else:
            self.tokenizer(data)
            dpoints+=len(data)
        print(f"Updated tokens from {dpoints} rows of data")
        self.save()
    def decode(self,data):
        return self.tokenizer.decode(data,skip_special_tokens=True,clean_up_tokenization_spaces=True,)


    


class Stream_Dataset(Dataset):
    def __init__(self,data_dir:str,max_seql:int=None,img_dir:str='',transforms=None,imgsz:int=300,csv_dir:str=None,tokenizer_model:Union[HFTokenizer,str]='medicalai/ClinicalBERT'):
        super(Stream_Dataset,self).__init__()
        self.data_dir=data_dir
        csv_dir=os.path.join(data_dir,csv_dir) if csv_dir else self.get_default_csv(data_dir)
        self.data=pd.read_csv(csv_dir).to_numpy()
        self.img_dir=img_dir
        self.imgsz=imgsz
        self.resizer=torchvision.transforms.Resize(size=(imgsz,imgsz))
        self.transforms=self.get_default_imgtransforms() if transforms=='default' else transforms
        tokenizer=HFTokenizer(tokenizer_model,max_length=max_seql) if isinstance(tokenizer_model,str) else tokenizer_model
        tokenizer.update(self.data[:,-1].tolist())
        del tokenizer
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self,idx):
        img_name=self.data[idx,0]
        semantic_str=self.data[idx,1].strip()
        img=self.resizer(torchvision.io.read_image(self.data_dir+'/'+self.img_dir+'/'+img_name))
        if self.transforms is not None:
            img=self.transforms(img)
        return img/255.,semantic_str
    
    def get_default_csv(self,data_dir):
        csv_list=glob.glob(data_dir+"/*.csv")
        largest_csv=sorted(csv_list,reverse=True,key=lambda x:len(pd.read_csv(x)))[0]
        return largest_csv
        
    
    def get_default_imgtransforms(self):
        transforms_=torch.nn.Sequential(
            transforms.RandomPosterize(bits=3,p=0.4),
            transforms.RandomAdjustSharpness(sharpness_factor=0.5),
            transforms.RandomRotation(degrees=(-10,20)),
            transforms.ColorJitter(brightness=.2, hue=.1),
            transforms.RandomAutocontrast()

        )
        return transforms_ 

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









#     from config import get_config
# class HFTokenizer:
#     def __init__(self,tokenizer_model,max_length):
#         tok_pth=os.path.join("data",tokenizer_model.split('/')[-1],"tokenizer")
#         self.name=tokenizer_model
#         self.tokenizer=torch.hub.load('huggingface/pytorch-transformers', 'tokenizer',tokenizer_model) if not os.path.exists(tok_pth) else torch.hub.load('huggingface/pytorch-transformers', 'tokenizer',tok_pth)
#         self.save_pth=tok_pth
#         if not max_length:
#             max_length=(get_config())['max_seql']
#         self.tokenizer.model_max_length=max_length
#         self.vocab=self.tokenizer.vocab
#     def __call__(self,data,padding=None,truncation=True,return_tensors='pt',add_special_tokens=True):
#         return self.tokenizer(data,padding=padding,truncation=truncation,return_tensors=return_tensors,add_special_tokens=add_special_tokens).input_ids
    
#     def decode(self,tok_ids):
#         return self.tokenizer.decode(tok_ids,skip_special_tokens=True,clean_up_tokenization_spaces=True)
    
#     def __len__(self):
#         return len(self.tokenizer)
    
#     def save(self):
#         self.tokenizer.save_pretrained(self.save_pth)
    
#     def update(self,data):
#         dpoints=0
#         if isinstance(data,torch.utils.data.dataloader.DataLoader):
#             for (img,text) in data:
#                 self.tokenizer(text)
#                 dpoints+=len(text)
#         elif isinstance(data,(np.ndarray,torch.Tensor)):
#             data=data.tolist()
#             self.tokenizer(data)
#             dpoints+=len(data)
#         else:
#             self.tokenizer(data)
#             dpoints+=len(data)
#         print(f"Updated tokens from {dpoints} rows of data")
#         self.save()

# class Stream_Dataset(Dataset):
#     def __init__(self,data_dir:str,max_seql:int=None,img_dir:str='',transforms=None,imgsz:int=300,csv_dir:str=None,tokenizer_model:Union[HFTokenizer,str]='medicalai/ClinicalBERT'):
#         self.data_dir=data_dir
#         csv_dir=os.path.join(data_dir,csv_dir) if csv_dir else self.get_default_csv(data_dir)
#         self.data=pd.read_csv(csv_dir).to_numpy()
#         self.img_dir=img_dir
#         self.imgsz=imgsz
#         self.resizer=torchvision.transforms.Resize(size=(imgsz,imgsz))
#         self.transforms=self.get_default_imgtransforms() if transforms=='default' else transforms
#         tokenizer=HFTokenizer(tokenizer_model,max_length=max_seql) if isinstance(tokenizer_model,str) else tokenizer_model
#         tokenizer.update(self.data[:,-1].tolist())
#         del tokenizer

#     def __len__(self):
#         return self.data.shape[0]
    
#     def __getitem__(self,idx):
#         img_name=self.data[idx,0]
#         semantic_str=self.data[idx,1].strip()
#         img=self.resizer(torchvision.io.read_image(self.data_dir+'/'+self.img_dir+'/'+img_name))
#         if self.transforms is not None:
#             img=self.transforms(img)
#         return img/255.,semantic_str
    
#     def get_default_csv(self,data_dir):
#         csv_list=glob.glob(data_dir+"/*.csv")
#         largest_csv=sorted(csv_list,reverse=True,key=lambda x:len(pd.read_csv(x)))[0]
#         return largest_csv
        
    
#     def get_default_imgtransforms(self):
#         transforms_=torch.nn.Sequential(
#             transforms.RandomPosterize(bits=3,p=0.4),
#             transforms.RandomAdjustSharpness(sharpness_factor=0.5),
#             transforms.RandomRotation(degrees=(-10,20)),
#             transforms.ColorJitter(brightness=.2, hue=.1),
#             transforms.RandomAutocontrast()

#         )
#         return transforms_ 