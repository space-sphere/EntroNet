__all__ = ['Entropy']

import torch
import torch.nn as nn
from layers.pTE import EntropyBackbone


class Model(nn.Module):
    def __init__(self, configs, activation='gelu', pe='zeros', head_type='flatten', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        head_individual = configs.head_individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        use_individual_entropy_value = configs.individual_entropy
        p_layers = configs.p_layers
        # use_individual_patch = configs.use_individual_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        n_heads_attn = configs.n_heads_attn
        d_model_attn = configs.d_model_attn
        
        self.decomposition = configs.decomposition
        self.kernel_size = configs.kernel_size
        
        lag = configs.lag
        model_order = configs.model_order
        output_attention = configs.output_attention
        
        # key_padding_mask, attn_dropout, 
        
        self.model = EntropyBackbone(c_in, context_window, target_window, patch_len, stride, d_model, n_heads,
                 mask_flag=False, dropout=dropout, lag=lag, model_order=model_order, output_attention=output_attention, d_ff=d_ff,
                 activation=activation, use_bias=False, e_layers=n_layers, revin=revin, affine=affine, subtract_last=subtract_last, padding_patch=padding_patch, 
                 pretrain_head=None, head_type=head_type, head_individual = head_individual, head_dropout=head_dropout, pe=pe, fc_dropout=fc_dropout,
                 p_layers=p_layers, n_heads_attn=n_heads_attn, d_model_attn=d_model_attn, use_individual_entropy_value=use_individual_entropy_value)
    
    def forward(self, x, attn_mask=None):
        if not self.decomposition:
            x = x.permute(0, 2, 1)                 # x: [bs, nvars, input_length]
            x = self.model(x, attn_mask)           # x: [bs, nvars, input_length]
            return x.permute(0, 2, 1)              # x: [bs, input_length, nvars]
        