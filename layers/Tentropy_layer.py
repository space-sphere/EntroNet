import torch
import torch.nn as nn
import numpy as np
from typing import Optional
import torch.nn.functional as F
from torch import Tensor
from utils.entropy import entro, transpose 
from layers.RevIN import RevIN
from layers.Entropy_layers import get_activation_fn, positional_encoding
from layers.Transformer_Layers import MultiheadAttention

class SingleTe(nn.Module):
    def __init__(self, d_entro, n_heads, d_forward, d_mutual, patch_len, patch_num, n_heads_forward, nvars, 
                 dropout=None, d_ff=256, store_attn=False, stride=1,
                 mutual_type='linear', mutual_individual=False,                                # mutual info
                 activation='gelu', res_attention=False, e_layers=1, lag=1, model_order=1,
                 head_individual=False, target_window=None,
                 padding_patch='start', revin=False, affine=False, subtract_last=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.res_attention = res_attention
        self.padding_patch = padding_patch
        self.patch_len = patch_len
        self.stride = stride
        
        # Revin Layer
        if revin:
            self.revin = revin
            self.revin_layer = RevIN(nvars, affine=affine, subtract_last=subtract_last)
        
        if self.padding_patch == 'start':
            self.padding_patch_layer = nn.ReplicationPad1d((stride, 0)) 
            patch_num += 1
        elif self.padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            patch_num += 1
            
        self.patch_num = patch_num
        head_nf = patch_num * d_forward
        
        # Encoder
        self.encoder = EntropyEncoder(
            d_entro=d_entro, n_heads=n_heads, d_forward=d_forward, d_mutual=d_mutual, patch_len=patch_len, patch_num=patch_num,
            n_heads_forward=n_heads_forward, nvars=nvars, dropout=dropout, d_ff=d_ff, store_attn=store_attn, mutual_type=mutual_type,
            mutual_individual=mutual_individual, activation=activation, res_attention=res_attention, e_layers=e_layers, lag=lag, model_order=model_order
        )
        
        # Decoder
        self.decoder = EntropyDecoder(
            individual=head_individual, nvars=nvars, head_nf=head_nf, target_window=target_window, head_dropout=dropout
        )
    
    def forward(self, z):                                                                   # [bs x seq_len x nvars]
        z = z.permute(0, 2, 1)                                                              # [bs x nvars x seq_len]
        
        if self.revin:
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)                                                            # [bs x nvars x seq_len]
        
        # Patch
        if self.padding_patch != None:
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
        
        # Encoder
        if self.res_attention:
            z, attn_scores, entropy_scores = self.encoder(z)
        else:
            z = self.encoder(z)

        # Decoder
        z = self.decoder(z)
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0, 2, 1)
        
        z = z.permute(0, 2, 1)                                                              # [bs, seq_len, nvars]
        
        if self.res_attention:
            return z, attn_scores, entropy_scores
        else:
            return z
        

class EntropyDecoder(nn.Module):
    def __init__(self, individual, nvars, head_nf, target_window, head_dropout=0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.individual = individual
        self.n_vars = nvars
        
        if not individual:
            self.flatten = nn.Flatten(-2)
            self.linear = nn.Linear(head_nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

        else:
            self.flatten = nn.ModuleList()
            self.linear = nn.ModuleList()
            self.dropout = nn.ModuleList()
            for var in range(nvars):
                self.flatten.append(nn.Flatten(-2))
                self.linear.append(nn.Linear(head_nf, target_window))
                self.dropout.append(nn.Dropout(head_dropout))
    
    def forward(self, x):
        """TEformer Decoder

        Args:
            x (Tensor)  : [bs, nvars, d_forward, patch_num]

        Returns:
            Tensor      : [bs, nvars, target_window]
        """
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flatten[i](x[:,i,:,:])          # z: [bs x d_model x patch_num]
                z = self.linear[i](z)                    # z: [bs x target_window]
                z = self.dropout[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x
 
class EntropyEncoder(nn.Module):
    def __init__(self, d_entro, n_heads, d_forward, d_mutual, patch_len, patch_num, n_heads_forward, nvars, 
                 dropout=None, d_ff=256, store_attn=False, 
                 mutual_type='linear', mutual_individual=False,                                # mutual info
                 activation='gelu', res_attention=False, e_layers=1, lag=1, model_order=1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.graph = EntropyGraph(patch_len=patch_len, n_hiddens=d_entro, n_heads=n_heads, dropout=dropout, lag=lag, model_order=model_order)
        self.entropy_dropout = nn.Dropout(dropout)
        
        self.res_attention = res_attention
        
        self.Wp = nn.Linear(patch_len, d_forward)
        
        # Entropy Sublayers
        self.entropy_layers = nn.ModuleList()
        for _ in range(e_layers):
            layer = EntropyEncoderLayer(d_forward, d_mutual, patch_num, n_heads_forward, nvars, 
                 dropout=dropout, d_ff=d_ff, store_attn=store_attn, 
                 mutual_type=mutual_type, mutual_individual=mutual_individual,                                # mutual info
                 activation=activation, res_attention=res_attention)
            
            self.entropy_layers.append(layer)
    
    def forward(self, z):                                                               # [bs, nvars, patch_num, patch_len]
        # generate entropy graph
        bs, nvars, patch_num, patch_len = z.shape
        entropy_scores = self.graph(z, z).view(bs, -1, nvars, nvars)                    # [bs, n_heads, nvars_q, nvars_v]
        graph_weight = self.entropy_dropout(F.softmax(entropy_scores, dim=-1))
        
        v = self.Wp(z)                                                                  # [bs, nvars, patch_num, d_forward]
        # Entropy Encoder Layer
        attn_scores = []
        for entropy_layer in self.entropy_layers:
            if self.res_attention:
                v, attn_score = entropy_layer(v, graph_weight)                          # [bs, nvars, patch_nu, d_forward]
                attn_scores.append(attn_score)
            else:
                v = entropy_layer(v, graph_weight)                                      # [bs, nvars, patch_nu, d_forward]
        
        if self.res_attention:
            return v, attn_scores, entropy_scores
        
        return v
        

class EntropyEncoderLayer(nn.Module):
    def __init__(self, d_forward, d_mutual, patch_num, n_heads_forward, nvars, 
                 dropout=None, d_ff=256, store_attn=False, 
                 mutual_type='linear', mutual_individual=False,                                # mutual info
                 activation='gelu', res_attention=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.res_attention = res_attention
        
        # CI Values
        self.image = CrossTimeMIP(d_forward, patch_num, n_heads_forward, d_ff=d_ff, store_attn=store_attn,
                                  attn_dropout=dropout, dropout=dropout, activation=activation, res_attention=res_attention)
        
        # Aggregate Info
        self.mutual = CrossDimMIP(n_heads=n_heads_forward, patch_num=patch_num, d_forward=d_forward,
                                 d_mutual=d_mutual, type=mutual_type, nvars=nvars, activation=activation, dropout=dropout, individual=mutual_individual)

        self.dropout_info = nn.Dropout(dropout)
        self.norm_info = nn.LayerNorm(d_forward)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_forward, d_ff),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_forward))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(d_forward)

    def forward(self, z, graph_weight):                                             # [bs, nvars, patch_num, d_forward]
        # CI in Attention Each Var
        if self.res_attention:
            crosstime, attn_scores = self.image(z)                                     # [bs, nvars, patch_num, d_forward]
        else:
            crossdime = self.image(z)
        
        # Aggregate Information
        output = self.mutual(z, graph_weight)                                       # [bs, nvars, patch_num, d_forward]
        
        output = (crosstime + crossdime) / 2
        output = self.norm_info(z + self.dropout_info(output))

        out = self.ff(output)
        output = self.norm_ffn(output + self.dropout_ffn(out))

        if self.res_attention:
            return output, attn_scores
        else:
            return output

class CrossDimMIP(nn.Module):
    def __init__(self, n_heads, patch_num, d_forward, d_mutual, type='linear', 
                 activation='gelu', nvars=None, individual=False, dropout=0, *args, **kwargs) -> None:
        """
        Same as MutualInfo
        """
        super().__init__(*args, **kwargs)
        
        assert d_forward % n_heads == 0, print('d_forward = {} is not able to be devided by n_heads = {}'.format(d_forward, n_heads))
        self.n_heads = n_heads
        
        assert type in ['linear', 'attention'], print('{} is not included in [linear, attention]'.format(type))
        self.type = type
        
        self.individual = individual
        self.dropout_entropy = nn.Dropout(dropout)
        
        if type == 'linear':
            
            if not individual:
                self.cross_sect = nn.Linear(d_forward // n_heads, d_forward // n_heads)
                self.cross_time = nn.Sequential(
                    nn.Linear(patch_num, d_mutual),
                    get_activation_fn(activation),
                    nn.Dropout(dropout),
                    nn.Linear(d_mutual, patch_num)
                )
                
            else:
                self.cross_sect = nn.ModuleList()
                self.cross_time = nn.ModuleList()
                for i in range(nvars):
                    cross_sect = nn.Linear(d_forward // n_heads, d_forward // n_heads)
                    cross_time = nn.Sequential(
                        nn.Linear(patch_num, d_mutual),
                        get_activation_fn(activation),
                        nn.Dropout(dropout),
                        nn.Linear(d_mutual, patch_num)
                    )
                    
                    self.cross_sect.append(cross_sect)
                    self.cross_time.append(cross_time)
        
        self.output_linear = nn.Linear(d_forward, d_forward)
        
    def forward(self, z, entropy):
        """aggregate each var by using gnn

        Args:
            values  (Tensor)    :    [bs, nvars, patch_num, d_forward]
            z       (Tensor)    :    [bs, nvars, patch_num, d_forward]
            entropy (Tensor)    :    [bs, n_heads, nvars_q, nvars_k ]

        Returns:
            Tensor              :    [bs, nvars, patch_num, d_forward]
        """
        bs, nvars, patch_num, d_forward = z.shape
        
        """use gnn to aggregate information"""
        if not self.individual:
            out = self.cross_sect(z.view(bs, nvars, patch_num, self.n_heads, -1)).permute(0, 3, 1, 4, 2)    # [bs, n_heads, nvars, d / h, patch_num]
            out = self.cross_time(out)                                                                      # [bs, n_heads, nvars, d / h, patch_num]
        
        else:
            out = []
            for i in range(nvars):
                u = z[:, i]                                                                                 # [bs, patch_num, d_forward]
                u = self.cross_sect[i](u.view(bs, patch_num, self.n_heads, -1)).permute(0, 2, 3, 1)         # [bs, heads, d / h, patch_nu]
                u = self.cross_time[i](u)                                                                   # [bs, heads, d / h, patch_nu]
                out.append(u)
            out = torch.stack(out, dim=2)                                                                   # [bs, n_heads, nvars, d / h, patch_num]
        out = out.permute(0, 1, 2, 4, 3)                                                                    # [bs, n_heads, nvars, patch_num, d / h]
        
        # Mutual Info
        out = torch.matmul(self.dropout_entropy(entropy), out.reshape(bs, self.n_heads, nvars, -1))         # [bs, n_heads, nvars, p * d / h]
        out = out.view(bs, self.n_heads, nvars, patch_num, d_forward // self.n_heads)
        out = out.permute(0, 2, 3, 1, 4).reshape(bs, nvars, patch_num, -1)                                  # [bs, nvars, patch_num, d_forward]
        # out = out + values
        out = self.output_linear(out)
        
        return out

class EntropyGraph(nn.Module):
    def __init__(self, patch_len, n_hiddens, n_heads, d_keys=None, dropout=0.1, lag=1, model_order=1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        d_keys = n_hiddens // n_heads
        self.n_heads = n_heads
        
        self.lag = lag
        self.model_order = model_order
        
        self.Wq = nn.Linear(patch_len, d_keys * n_heads)
        self.Wk = nn.Linear(patch_len, d_keys * n_heads)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, queries, keys, eps=1e-6):                                      # [bs, nvars, patch_num, patch_len]
        # TODO make sure keys == torch.cat([queries, exo], dim=1)
        
        qt = transpose(self.Wq(queries), self.n_heads)                               # [bs * heads, nvars, d_model / heads, patch_num]
        kt = transpose(self.Wk(keys), self.n_heads)                                  # [bs * heads, nvars, d_model / heads, patch_num]
        
        entropy = entro(qt, kt, eps, self.lag, self.model_order)                     # [bs * heads, nvars_q, nvars_k] TE_k->q
        
        indices = torch.arange(queries.shape[1], device=queries.device).view(-1, 1)
        entropy[:, indices, indices] = -np.inf
        
        return entropy

class CrossTimeMIP(nn.Module):
    def __init__(self, d_forward, patch_num, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 attn_dropout=0, dropout=0., activation="gelu", res_attention=False, pe='zeros',
                 learn_pe=False, *args, **kwargs) -> None:
        """
        Same as GraphForward
        """
        super().__init__(*args, **kwargs)
        self.res_attention = res_attention
        
        
        # Positional encoding
        self.Wpos = positional_encoding(pe, learn_pe, patch_num, d_forward)
        
        # Residual dropout
        self.dropout = nn.Dropout(dropout)
        
        # Graph Sublayer
        self.graph_layer = GraphForwardLayer(d_forward, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, store_attn=store_attn,
                 attn_dropout=attn_dropout, dropout=dropout, activation=activation, res_attention=res_attention)
    
    def forward(self, z):                                                             # [bs, nvars, patch_num, d_forward]
        nvars = z.shape[1]
        
        z = z.view(-1, z.shape[2], z.shape[3])                                        # [bs * nvars, patch_num, d_forward]
        z = self.dropout(z + self.Wpos)
        
        if self.res_attention:
            z, scores = self.graph_layer(z)
            z = z.view(-1, nvars, z.shape[1], z.shape[2])                             # [bs, nvars, patch_num, d_forward]
            return z, scores
        
        z = self.graph_layer(z)
        z = z.view(-1, nvars, z.shape[1], z.shape[2])
        return z
        
class GraphForwardLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 attn_dropout=0, dropout=0., activation="gelu", res_attention=False):
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = nn.LayerNorm(d_model)

        # # Position-wise Feed-Forward
        # self.ff = nn.Sequential(nn.Linear(d_model, d_ff),
        #                         get_activation_fn(activation),
        #                         nn.Dropout(dropout),
        #                         nn.Linear(d_ff, d_model))

        # # Add & Norm
        # self.dropout_ffn = nn.Dropout(dropout)
        # self.norm_ffn = nn.LayerNorm(d_model)

        self.store_attn = store_attn


    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:
        # Multi-Head attention sublayer     # [bs * nvars, patch_num, d_model]
        
        ## Multi-Head attention
            
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        # Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        src = self.norm_attn(src)

        # Feed-forward sublayer
        
        ## Position-wise Feed-Forward
        # src2 = self.ff(src)
        
        ## Add & Norm
        # src = src + self.dropout_ffn(src2)  # Add: residual connection with residual dropout
        # src = self.norm_ffn(src)
        
        if self.res_attention:
            return src, scores
        else:
            return src
