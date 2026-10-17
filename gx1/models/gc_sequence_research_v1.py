"""Small GC-only research encoders, never registered as native Entry/Exit."""
import torch
from torch import nn
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import _sequence_readout


class GCSequenceTransformer(nn.Module):
    def __init__(self,features,length,recipe):
        super().__init__();width=recipe['width']
        self.proj=nn.Linear(features,width)
        self.position=nn.Parameter(torch.empty(1,length,width))
        nn.init.normal_(self.position,std=recipe['position_init_std'])
        layer=nn.TransformerEncoderLayer(width,recipe['heads'],recipe['feedforward'],
            dropout=recipe['dropout'],activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,recipe['layers'],enable_nested_tensor=False)
        self.out=nn.Linear(width,1);self.length=length;self.features=features

    def forward(self,x):
        if x.ndim!=3 or x.shape[1:]!=(self.length,self.features) or not torch.isfinite(x).all():
            raise ValueError('GC_SEQUENCE_TRANSFORMER_INPUT')
        # Entire input is closed by the decision; positions retain ordering.
        return self.out(_sequence_readout(self.encoder(self.proj(x)+self.position))).squeeze(-1)


class GCSequenceTCN(nn.Module):
    def __init__(self,features,length,recipe):
        super().__init__();width=recipe['width'];kernel=recipe['kernel']
        self.proj=nn.Conv1d(features,width,1);self.length=length;self.features=features
        self.convs=nn.ModuleList([nn.Conv1d(width,width,kernel,dilation=d) for d in recipe['dilations']])
        self.norms=nn.ModuleList([nn.LayerNorm(width) for _ in self.convs])
        self.out=nn.Linear(width,1)
        self.receptive_field=1+(kernel-1)*sum(recipe['dilations'])
        if self.receptive_field>length:raise ValueError('GC_SEQUENCE_TCN_RECEPTIVE_FIELD')

    def tokens(self,x):
        z=self.proj(x.transpose(1,2))
        for conv,norm in zip(self.convs,self.norms):
            # Valid causal convolution; no synthetic zero-filled book history.
            y=conv(z);z=z[:,:,-y.shape[-1]:]+torch.nn.functional.gelu(norm(y.transpose(1,2))).transpose(1,2)
        return z.transpose(1,2)

    def forward(self,x):
        if x.ndim!=3 or x.shape[1:]!=(self.length,self.features) or not torch.isfinite(x).all():
            raise ValueError('GC_SEQUENCE_TCN_INPUT')
        return self.out(_sequence_readout(self.tokens(x))).squeeze(-1)
