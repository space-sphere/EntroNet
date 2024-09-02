import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
import numpy as np
from utils.entropy import TransferEntropy
from layers.Entropy_layers import get_activation_fn, positional_encoding
from layers.PatchTST_backbone import TSTEncoder
from layers.RevIN import RevIN


class EntropyBackbone(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, d_model, n_heads,
                 mask_flag=False, n_heads_attn=8,  dropout=0.1, lag=1, model_order=1, output_attention=False, d_ff=None,
                 activation='gelu', use_bias=False, e_layers=None, revin=True, affine=True, subtract_last=False, padding_patch='end', 
                 pretrain_head=None, head_type='flatten', head_individual = False, fc_dropout=0, head_dropout=0, pe='zeros', learn_pe=True, p_layers=1,
                 use_attention=True, d_model_attn = 512, use_individual_entropy_value = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        self.revin = revin
        
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.use_attention = use_attention
        self.n_vars = c_in
        self.padding_patch = padding_patch
        
        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end':  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((stride, 0)) 
            patch_num += 1
        
        if use_individual_entropy_value:
            n_vars = self.n_vars
        else:
            n_vars = None
        
        # iEncoder
        self.iencoder = EntropyiEncoder(d_model, n_heads, mask_flag=mask_flag, dropout=dropout, lag=lag, model_order=model_order, output_attention=output_attention,
                 d_ff=d_ff, activation=activation, use_bias=use_bias, e_layers=e_layers, patch_len=patch_len, patch_num=patch_num, pe=pe, learn_pe=learn_pe, n_vars=n_vars)
        
        # PatchEncoder
        if self.use_attention:
            self.patch_iencoder = PatchiEncoder(d_model, d_model_attn, n_heads_attn=n_heads_attn, d_k=None, d_v=None, d_ff=d_ff, norm='laye', attn_dropout=dropout, dropout=dropout,
                                    pre_norm=False, activation=activation, res_attention=output_attention, n_layers=p_layers, store_attn=False)
            d_model = d_model_attn
        
        # Head
        self.output_attention = output_attention
        self.head_nf = d_model * patch_num
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.head_individual = head_individual
        
        if head_type == 'flatten':
            self.head = FlattenHead(head_individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
    
    def forward(self, z, attn_mask=None):                                                   # x: [bs, nvars, seq]
        # norm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)
            
        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0,1,3,2)                                                              # z: [bs x nvars x patch_len x patch_num]
        
        # model
        z, entropy = self.iencoder(z, attn_mask)                                            # z: [bs, nvars, d_model, patch_num]
        if self.use_attention:
            z = self.patch_iencoder(z)                                                      # z:[bs, nvars, d_model_attn, p]
        z = self.head(z)
        
        # denorm
        if self.revin:
            z = z.permute(0, 2, 1)                                                          # z: [bs, nvars, target_window] - [bs, target_window, nvars]
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0, 2, 1)
        
        if self.output_attention:
            return z, entropy
        return z

class PatchiEncoder(nn.Module):
    def __init__(self, d_model, d_model_attn, n_heads_attn, d_k=None, d_v=None, d_ff=2048, norm='laye', attn_dropout=0.1, dropout=0.1,
                                    pre_norm=False, activation='gelu', res_attention=False, n_layers=1, store_attn=False,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear = nn.Linear(d_model, d_model_attn)
        self.patch_encoder = TSTEncoder(0, d_model_attn, n_heads=n_heads_attn, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                    pre_norm=pre_norm, activation=activation, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)
    
    def forward(self, x):                                                   # [bs, nvars, d_model, p_num]
        x = self.linear(x.permute(0, 1, 3, 2))                              # [bs, nvars, p_num, d_model_attn]
        nvars = x.shape[1]
        x = x.reshape(-1, x.shape[2], x.shape[3])                           # [bs nvars, p, d]
        x = self.patch_encoder(x.reshape(-1, x.shape[-2], x.shape[-1]))     # [bs nvars, p_num, d_model_attn]
        x = x.reshape(-1, nvars, x.shape[-2], x.shape[-1])                  # [bs, nvars, p_num, d_model_attn]
        return x.permute(0, 1, 3, 2)                                        # [bs, nvars, d, p]
        

class EntropyiEncoder(nn.Module):
    def __init__(self, d_model, n_heads, mask_flag=False, dropout=0.1, lag=1, model_order=1, output_attention=False,
                 d_ff=None, activation='gelu', use_bias=False, e_layers=None, patch_num=None, patch_len=None,
                 pe='zeros', learn_pe=True, n_vars=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        # Input encoding
        self.Wp = nn.Linear(patch_len, d_model, bias=use_bias)
        
        # Positional encoding
        self.Wpos = positional_encoding(pe, learn_pe, patch_num, d_model)
        
        # Residual dropout
        self.dropout = nn.Dropout(dropout)
        
        # Encoder
        self.encoder = EntropyEncoder(d_model, n_heads, mask_flag=mask_flag, dropout=dropout, lag=lag, model_order=model_order, output_attention=output_attention,
                 d_ff=d_ff, activation=activation, use_bias=use_bias, e_layers=e_layers, n_vars=n_vars)
        
    def forward(self, x, attn_mask):                                      # [batch_size, nvars, patch_len, patch_num]
        nvars = x.shape[1] 
        x = x.permute(0, 1, 3, 2)                                         # [batch_size, nvars, patch_num, patch_len]
        
        x = self.Wp(x)                                                    # [bs, nvars, patch_num, d_model]
        x = self.dropout(x + self.Wpos)
        
        # Encoder
        x, entropy_list = self.encoder(x, attn_mask)                      # [bs, nvars, patch_num, d_model]
        x = x.permute(0, 1, 3, 2)
        
        return x, entropy_list                                            # [bs, nvars, d_model, patch_num]

class EntropyEncoder(nn.Module):
    def __init__(self, d_model, n_heads, mask_flag=False, dropout=0.1, lag=1, model_order=1, output_attention=False,
                 d_ff=None, activation='gelu', use_bias=False, e_layers=None, n_vars=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder_layers = nn.ModuleList(
            [EntropyEncoderLayer(
                d_model, n_heads, mask_flag=mask_flag, dropout=dropout, lag=lag, model_order=model_order, output_attention=output_attention,
                 d_ff=d_ff, activation=activation, use_bias=use_bias, n_vars=n_vars
            ) for i in range(e_layers)]
        )
        
    def forward(self, src, attn_mask):
        output = src
        entropy_list = []
        for mod in self.encoder_layers:
            output, entropy = mod(output, attn_mask)
            entropy_list.append(entropy)
        return output, entropy_list

class EntropyEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, mask_flag=False, dropout=0.1, lag=1, model_order=1, output_attention=False,
                 d_ff=None, activation='gelu', use_bias=False, n_vars=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fullentropy = Full_Entropy(TransferEntropy(
                mask_flag=mask_flag, dropout=dropout, lag=lag, model_order=model_order, output_attention=output_attention, n_vars=n_vars
            ), n_hiddens=d_model, n_heads=n_heads)
        self.ffn = PositionWiseFFN(d_model, d_ff, d_model, activation, use_bias)
        self.dropout_entro = nn.Dropout(dropout)
        self.norm_entro = nn.LayerNorm(d_model)
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(d_model)
    
    def forward(self, src, attn_mask):                                            # [batch_size, feature_num, patch_num, d_model]
        src1, entropy = self.fullentropy(src, src, src, attn_mask)
        # add norm  
        src1 = src + self.dropout_entro(src1)
        src1 = self.norm_entro(src1)
        # position wise ffn
        src2 = self.ffn(src1)
        # add norm
        src2 = src1 + self.dropout_ffn(src2)
        return self.norm_ffn(src2), entropy
        
class FlattenHead(nn.Module):
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
        """_summary_

        Args:
            x (tensor): [bs, nvars, d_model, patch_num]

        Returns:
            tensor: [bs, nvars, target_window]
        """
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flatten[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linear[i](z)                    # z: [bs x target_window]
                z = self.dropout[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x
        
        
class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, activation, bias, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens, bias=bias)
        self.activation = get_activation_fn(activation=activation)
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs, bias=bias)
    
    def forward(self, X):
        return self.dense2(self.activation(self.dense1(X)))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights

class MultiHeadTransferEntropy(nn.Module):
    def __init__(self, entropy, query_size, key_size, value_size, num_hiddens, num_heads, use_bias=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.Wq = nn.Linear(query_size, num_hiddens, bias=use_bias)
        self.Wk = nn.Linear(key_size, num_hiddens, bias=use_bias)
        self.Wv = nn.Linear(value_size, num_hiddens, bias=use_bias)
        self.Wo = nn.Linear(num_hiddens, num_hiddens, bias=use_bias)
        self.entropy = entropy
    
    def transpose(self, X, num_heads):
        # shape: [b, f, d, t] [b, f, patch_len, patch_num]
        X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
        X = X.permute(0, 2, 1, 3)
        
        return X.reshape(-1, X.shape[2], X.shape[3])
    
    def transpose_output(self, X, num_heads):
        dshape = X.shape
        X = X.reshape(-1, self.num_heads, dshape[1], dshape[2])
        return X.permute(0, 2, 1, 3).reshape(-1, dshape[1], num_heads * dshape[2])
    
    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        queries = self.transpose(self.Wq(queries), self.num_heads)
        keys = self.transpose(self.Wk(keys), self.num_heads)
        values = self.transpose(self.Wv(values), self.num_heads)
        
        output, entropy = self.entropy(queries, keys, values, attn_mask)
        return self.Wo(self.transpose_output(output, self.num_heads)), entropy

class Full_Entropy(nn.Module):
    def __init__(self, entropy, n_hiddens, n_heads, d_keys=None, d_values=None, dropout=0.1, use_bias=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        d_keys = d_keys or (n_hiddens // n_heads)
        d_values = d_values or (n_hiddens // n_heads)
        self.n_heads = n_heads
        self.Wq = nn.Linear(n_hiddens, d_keys * n_heads, bias=use_bias)
        self.Wk = nn.Linear(n_hiddens, d_keys * n_heads, bias=use_bias)
        self.Wv = nn.Linear(n_hiddens, d_values * n_heads, bias=use_bias)
        self.Wo = nn.Linear(d_values * n_heads, n_hiddens, bias=use_bias)
        self.dropout = nn.Dropout(dropout)
        self.entropy = entropy
    
    def transpose(self, X, n_heads):                                          # [batch_size, nvars, patch_num, d_model]
        # [b, f, t, d]
        X = X.reshape(X.shape[0], X.shape[1],X.shape[2] , n_heads, -1)        # [b, f, t, h, d / h]
        X = X.permute(0, 3, 1, 4, 2)                                          # [b, h, f, d/h, t]
        return X.reshape(-1, X.shape[2], X.shape[3], X.shape[4])              # [-1, f, d/h, t]
    
    def transpose_output(self, X, n_heads):
        # [-1, f, d / h, t]
        dshape = X.shape
        X = X.reshape(-1, n_heads, dshape[1], dshape[2], dshape[3])           # [b, h, f, d / h, t]
        # [b, f, h, d / h, t]
        return X.permute(0, 2, 4,1, 3).reshape(-1, dshape[1], dshape[3], n_heads * dshape[2])  # [b, f, t, d]
    
    def forward(self, queries, keys, values, attn_mask, n_vars=None):
        """_summary_

        Args:
            queries   (tensor)      : shape: [batch_size, feature_num, patch_num, d_model]
            keys      (tensor)      : shape: [batch_size, feature_num, patch_num, d_model]
            values    (tensor)      : shape: [batch_size, feature_num, patch_num, d_model]
            attn_mask (bool)        : whether to use mask
        """
        queries = self.transpose(self.Wq(queries), self.n_heads)
        keys = self.transpose(self.Wk(keys), self.n_heads)
        values = self.transpose(self.Wv(values), self.n_heads)                # [batch_size * head, nvars, d_model / head, patch_num]
        
        output, entropy = self.entropy(queries, keys, values, attn_mask)      # [batch_size * head, nvars, d_model / head, patch_num]
        
        return self.Wo(self.transpose_output(output, self.n_heads)), entropy
        