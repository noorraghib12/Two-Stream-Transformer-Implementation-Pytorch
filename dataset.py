import torchvision.io
import torch
from torch.utils.data import Dataset

def load_tokenizer(tokenizer_model:str,data_dir:str=None):
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



class Stream_Dataset(Dataset):
    def __init__(self,seq_length,data_dir,img_dir,tokenizer_model='medicalai/ClinicalBERT'):
        self.data_dir=data_dir
        # self.data=pd.read_csv(data_dir+'/'+'result0.csv').to_numpy()
        self.img_dir=img_dir
        self.max_length=seq_length
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self,idx):
        img_name=self.data[idx,0]
        semantic_str=self.data[idx,1]
        img=torchvision.io.read_image(self.data_dir+'/'+self.img_dir+'/'+img_name)
        return img,semantic_str


if __name__=='__main__':
    # dataset=Stream_Dataset(
    #         seq_length=50,
    #         data_dir="/home/mehedi/Desktop/raghib/FLICKR-30K IMAGE CAPTIONING/flickr30k_images",
    #         img_dir='flickr30k_images',
    #         tokenizer_model='bert-base-uncased'
    # )
    pass