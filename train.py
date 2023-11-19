import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.text import WordErrorRate
import argparse
from config import Config
import datetime
from dataset import HFTokenizer,Stream_Dataset
from torch.optim import Adam,SGD, RMSprop
from torch.nn import CrossEntropyLoss
from model import TwoStreamTransformer
from lightning.fabric import Fabric
import torch
import torchmetrics
from torch.utils.data import Dataset, DataLoader
from utils import *



arg_parser=argparse.ArgumentParser()
arg_parser.add_argument('--config','-c',help='Path to config file',default='config.ini',type=str,required=False)
args=arg_parser.parse_args()


arg_parser=argparse.ArgumentParser()
arg_parser.add_argument('--config','-c',help='Path to config file',default=None,type=str)
args=arg_parser.parse_args()

config=Config(config_path=args.config)
 
fabric=Fabric(accelerator=config.device,devices=config.devices_n,precision=config.precision)
fabric.launch()


tokenizer=HFTokenizer(tokenizer_model=config.tokenizer_model)
training_set=Stream_Dataset(
    data_dir='flikr8k',
    csv_dir='train.csv',
    img_dir='images',
    imgsz=config.imgsz,
    tokenizer_model=config.tokenizer_model
)
validation_set=Stream_Dataset(
    data_dir='flikr8k',
    csv_dir='train.csv',
    img_dir='images',
    imgsz=config.imgsz,
    tokenizer_model=config.tokenizer_model,
    validation=True
)

training_loader=DataLoader(training_set,batch_size=config.batch_size)
validation_loader=DataLoader(validation_set,batch_size)


optimizer=load_optimizer(config.optimizer)
loss_fn=CrossEntropyLoss()
accuracy_fn=torchmetrics.classification.Accuracy(task='multiclass',num_classes=len(tokenizer))



model=TwoStreamTransformer(n_patches=config.n_patches,
                           img_chw=(3,config.imgsz,config.imgsz),
                           max_seql=config.max_seql,
                           num_ts_blocks=config.num_ts_blocks,
                           n_head=config.nhead,
                           dropout=config.dropout,
                           d_model=config.d_model,
                           dim_feedforward=config.dim_feedforward,
                           activation=config.activation,
                           tokenizer_model=config.tokenizer_model)







model,optimizer=fabric.setup(model,optimizer)
training_loader,validation_loader=fabric.setup_dataloaders([training_loader,validation_loader])
num_steps = config.num_epochs * len(training_loader)



scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_steps)


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.
    running_acc=0.
    last_acc=0.



    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + caption pair
        images, captions = data
        caption_idx=tokenizer(captions,padding='max_length',return_tensors='pt')
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs,attn=model(tgt=captions,image=images)
        # Compute the loss and its gradients
        loss = loss_fn(outputs, caption_idx)
        acc_=accuracy_fn(outputs,caption_idx)

        fabric.backward(loss)
        
        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_acc+=acc_.item()
        running_loss += loss.item()
        if i % 50 == 49:
            last_loss = running_loss / 50 # loss per
            last_acc = running_acc / 50
            print('batch {} train_loss: {} train_acc: {}'.format(i + 1, last_loss,last_acc))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            tb_writer.add_scaler('Accuracy/train',last_acc,tb_x)
            running_loss = 0.
            running_acc = 0.

    return last_loss,last_acc



def Trainer(epochs,model,track_opt):


    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/TSTransformer_trainer_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = epochs
    best_vloss = 1_000_000.
    best_vacc = .50 
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train()
        avg_loss,avg_acc = train_one_epoch(epoch_number, writer)


        running_vloss = 0.0
        running_vacc=0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                vlabels_idx=tokenizer(vlabels,padding='max_length',return_tensors='pt')
                voutputs,attn=model(images=vinputs)
                vloss = loss_fn(voutputs.transpose(1,2), vlabels_idx)
                vacc = accuracy_fn(voutputs.transpose(1,2),vlabels_idx)
                running_vloss += vloss
                running_vacc +=vacc

        avg_vloss = running_vloss / (i + 1)
        avg_vacc = running_vacc /(i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        print('ACCURACY train {} valid {}'.format(avg_acc,avg_vacc))
        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.add_scalars('Training vs. Validation Accuracy',
                        { 'Training' : avg_acc, 'Validation' : avg_vacc },
                        epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            m_path = 'runs/model_{}_{}'.format(timestamp, epoch_number)
            if track_opt.lower()=='loss':
                torch.save(model.state_dict(), m_path)
        if avg_vacc< best_vacc:
            best_vacc=avg_vacc
            m_path = 'runs/model_{}_{}'.format(timestamp, epoch_number)
            if track_opt.lower()==('accuracy' or 'acc'):
                torch.save(model.state_dict(), m_path)
            

        epoch_number += 1

if __name__=='__main__':
    Trainer(epochs=25,model=model,track_opt='acc')


