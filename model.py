from typing import Optional, Any, Union, Callable, Tuple
import torch
from torch import math, Tensor
import torch.nn as nn
from torch.nn import (
    MultiheadAttention,
    LayerNorm,
    Dropout,
    Linear,
    TransformerEncoderLayer,
    functional as F,
)
from torch.utils.data import Dataset, DataLoader
from utils import *

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = 1 / (10000 ** (torch.arange(0, d_model, 2) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, patches):
        return self.dropout(patches + self.pe)


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class CustomTransformerDecoderLayer(nn.Module):
    __constants__ = ["batch_first", "norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        need_attn: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )
        self.multihead_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation
        self.need_attn = need_attn

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as tgt mask.
                Mutually exclusive with providing tgt_mask. Default: ``False``.
            memory_is_causal: If specified, applies a causal mask as tgt mask.
                Mutually exclusive with providing memory_mask. Default: ``False``.
        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal
            )
            x = x + self._mha_block(
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
                memory_is_causal,
            )
            x = x + self._ff_block(self.norm3(x))
        else:
            x_sa = self.norm1(
                x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            )
            x, attn = self._mha_block(
                x,
                memory,
                memory_mask,
                memory_key_padding_mask,
                memory_is_causal,
            )
            x = self.norm2(x_sa + x)
            x = self.norm3(x + self._ff_block(x))

        return x, attn

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x, attn = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            average_attn_weights=self.need_attn,
        )

        return self.dropout2(x), attn

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class ImgPatcher(nn.Module):
    def __init__(self,n_patches:int,patch_dims:int,img_chw:Tuple[int,int,int]):
        super(ImgPatcher,self).__init__()
        C,H,W=img_chw
        assert torch.math.ceil(torch.math.sqrt(n_patches))-torch.math.sqrt(n_patches)==0, "Please make sure n_patches is a square of a positive integer"
        assert H==W, "Please make sure that the image is a square image"
        self.n_patches=n_patches
        self.patch_dims=patch_dims
        self.kernel_size=int(H//torch.math.sqrt(n_patches))
        self.conv=nn.Conv2d(in_channels=C,out_channels=self.patch_dims,kernel_size=self.kernel_size,stride=self.kernel_size)
    def forward(self,x):
        x=self.conv(x)
        x=x.permute(0,2,3,1)
        x_emb=x.reshape(x.size(0),self.n_patches,x.size(-1))
        return x_emb


class SpatialStream(nn.Module):
    def __init__(self, n_patches:int,img_chw:Tuple[int,int,int],dropout:float=0.1,d_model:int=768,nhead:int=8,dim_feedforward:int=2048,activation:str='relu',num_ts_blocks:int=2):
        super(SpatialStream,self).__init__()
        self.patcher=ImgPatcher(n_patches,d_model,img_chw)
        self.pos_enc=PositionalEncoding(d_model,dropout,n_patches)
        self.spattransformer_blocks=nn.ModuleList([TransformerEncoderLayer(d_model,nhead,dim_feedforward,batch_first=True,activation=activation) for i in range(num_ts_blocks)])
        
    def forward(self,x):
        embedded=self.patcher(x)
        encoded=self.pos_enc(embedded)
        enc_memory_list=[]
        for encoder_layer in self.spattransformer_blocks:
            encoded=encoder_layer(encoded)
            enc_memory_list.append(encoded)
        return enc_memory_list

class SemanticStream(nn.Module):
    def __init__(self,max_seql:int,dropout:float=0.1,num_ts_blocks:int=2,d_model:int=512,nhead:int=8,dim_feedforward:int=2048,activation:str='relu',tokenizer_model='medicalai/ClinicalBERT'):
        super(SemanticStream,self).__init__()
        self.max_seql=max_seql
        self.tokenizer=load_tokenizer(tokenizer_model=tokenizer_model)            #add this to dataloader instead of model along with patching
        self.bert_embedding = load_bert_embedder(tokenizer_model=tokenizer_model,tokenizer=self.tokenizer)
        self.pos_enc=PositionalEncoding(d_model,dropout,max_seql)
        self.semtransformer_blocks=nn.ModuleList([CustomTransformerDecoderLayer(d_model,nhead,dim_feedforward,batch_first=True,activation=activation) for _ in range(num_ts_blocks)])
        self.out=nn.Linear(d_model,len(tokenizer))
    def forward(self,tgt,enc_memory_list):
        tgt=self.tokenizer(tgt,max_length=self.max_seql,padding='max_length',truncation=True,return_tensors='pt')
        tokens=tgt['input_ids']
        tgt_padmask=tgt['attention_mask']==0
        #N,max_seql=tgt['input_ids'].shape
        tgt_mask=torch.tril(torch.ones(self.max_seql,self.max_seql))
        embedded=self.bert_embedding(**tgt).last_hidden_state
        encoded=self.pos_enc(embedded)
        for decoder_layer,memory in zip(self.semtransformer_blocks,enc_memory_list):
            encoded,attn=decoder_layer(encoded,memory,tgt_mask=tgt_mask,tgt_key_padding_mask=tgt_padmask)
            tgt_mask=None
            tgt_padmask=None
        out=self.out(encoded)

        return out,attn

    def save_bert_embedder(self):
        self.bert_embedding.save_pretrained('data/model/')


# class TwoStreamTransformer(nn.Module):
#     def __init__(self,max_seql,dropout)