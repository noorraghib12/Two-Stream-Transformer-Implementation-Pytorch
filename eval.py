from torch import nn
import torch
from torchmetrics.text import BLEUScore, ROUGEScore,WordErrorRate
from torchmetrics.aggregation import MeanMetric
from dataset import Stream_Dataset,HFTokenizer
from config import load_config_ini, get_config, Config
from torch.utils.data import Dataset,DataLoader
import argparse
from model import TwoStreamTransformer



arg_parser=argparse.ArgumentParser()
arg_parser.add_argument('--config','-c',help='Path to config file',type=str,required=True)
arg_parser.add_argument('--tokenizer_name','-tok',help='Path to config file',default=None,type=str,required=False)
arg_parser.add_argument('--model_path','-mp',help='Path to model file',required=True,type=str,required=False)
arg_parser.add_argument('--batch_size','-bsz',help='Dataloader Batch Size',default=None,type=int,required=False)
arg_parser.add_argument('--device','-d',help='device to compute with',default=None,type=str,required=False)

args=arg_parser.parse_args()

config_path=args.config
config=Config(config_path=config_path)

tokenizer_model=args.tokenizer_model if args.tokenizer_model else config.tokenizer_model


tokenizer=HFTokenizer(tokenizer_model=args.tokenizer_name)
rouge_l=ROUGEScore(rouge_keys=['rougeL'])
bleu4=BLEUScore(n_grams=4)
wer=WordErrorRate()

device=torch.device(device=args.device if args.device else config.device)

dataset=Stream_Dataset(
    data_dir=config.data_dir,
    img_dir=config.img_dir,
    csv_dir=config.csv_dir,
    tokenizer_model=tokenizer_model
)

dataloader=DataLoader(dataset,batch_size=args.batch_size if args.batch_size else config.val_size)
def evaluate(model,dataloader):
    model.eval()
    avg_rougeLScore=MeanMetric()
    avg_bleu4Score=MeanMetric()
    avg_wer=MeanMetric()
    
    with torch.no_grad():
        for data in dataloader:
            images, captions_str=data
            output,attn=model(images=images)
            output_probs=F.log_softmax(output,axis=-1)
            output_idx=output_probs.topk(1)[1]
            output_str=tokenizer.decode(output_idx)
            rougeLScore=rouge_l(output_str.tolist(),captions_str)
            bleu4Score=bleu4(output_str.tolist(),captions_str)
            werScore=wer(output_str.tolist(),captions_str)
            avg_rougeLScore.update(rougeLScore)
            avg_bleu4Score.update(bleu4Score)
            avg_wer.update(werScore)
        print(
            f"Summary Evaluation Metrics: \
            \n\t\t BLEU4:{(avg_bleu4Score.compute()).item()},\n\t\t ROUGEL:{(avg_rougeLScore.compute()).item()},\n\t\t WORD_ERROR_RATE:{(avg_wer.compute()).item()},"
        )

            
if __name__=='__main__':
    
    
    dataloader=DataLoader(dataset,batch_size=args.val_size).to(device)
    model=(torch.load(args.model_path)).to(device)
    evaluate(model,dataloader)
