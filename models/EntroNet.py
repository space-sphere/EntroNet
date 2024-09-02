import torch
import torch.nn as nn
from layers.EntroNet_layer import SingleTe

class Model(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        
        # Revin
        nvars = configs.nvars
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        # Patch
        stride_list = configs.stride
        if isinstance(stride_list, int):
            stride_list = [stride_list]
        stride_list.sort()
        seq_len = configs.seq_len
        patch_len_list = configs.patch_len
        if isinstance(patch_len_list, int):
            patch_len_list = [patch_len_list]
        patch_len_list.sort()
        patch_num = int((seq_len - patch_len_list[-1]) / stride_list[-1] + 1)
        
        padding_patch = configs.padding_patch
        
        # Encoder
        lag = configs.lag
        model_order = configs.model_order
        fast = bool(configs.use_fast), 
        # d_entro = configs.d_entro
        n_heads = configs.n_heads

        d_model = configs.d_model
        d_forward = d_model
        d_entro = d_forward
        d_mutual = configs.d_mutual
        
        n_heads_forward = configs.n_heads_forward
        dropout = configs.dropout
        d_ff = configs.d_ff
        store_attn = False
        mutual_type = configs.mutual_type
        mutual_individual = configs.mutual_individual
        activation = configs.activation
        res_attention = configs.res_attention
        e_layers = configs.e_layers
        
        self.lenth = len(patch_len_list)
        # self.stride = stride
        self.res_attention = res_attention
        
        # self.use_multiscale = configs.use_multiscale
        # kernel_sizes = configs.kernel_sizes
        
        # Decoder
        individual = configs.head_individual
        target_window = configs.pred_len

        if len(patch_len_list) > 1:
            # self.agg = nn.Conv1d(in_channels=len(patch_len_list), out_channels=1, kernel_size=1)
            self.agg = nn.AvgPool1d(3)
        
        self.model = nn.ModuleList()
        for i in range(len(patch_len_list)):
            patch_len = patch_len_list[i]
            stride = stride_list[i]

            model = SingleTe(
                d_entro=d_entro, n_heads=n_heads, d_forward=d_forward, d_mutual=d_mutual, patch_len=patch_len, patch_num=patch_num,
                n_heads_forward=n_heads_forward, nvars=nvars, dropout=dropout, d_ff=d_ff, store_attn=store_attn, stride=stride, mutual_type=mutual_type,
                mutual_individual=mutual_individual, activation=activation, res_attention=res_attention, e_layers=e_layers,lag=lag,
                model_order=model_order, head_individual=individual, target_window=target_window,
                padding_patch=padding_patch, revin=revin, affine=affine, subtract_last=subtract_last, fast=fast, 
            )
            self.model.append(model)
    
    def forward(self, z):                                                                   # [bs, seq_len, nvars]
        # whether to use multi-scale
        v_list = []
        for model in self.model:
            if self.res_attention:
                v, self.attn_scores, self.entropy_scores = model(z)
            
            else:
                v = model(z)
            v_list.append(v)

        if len(v_list) == 1:

            return v.permute(0, 2, 1)
        
        v_list = torch.stack(v_list, dim=2)                                                  # [bs, nvars, pl, seq_len]
        bs, nvars, pl, seq_len = v_list.shape
        
        v_list = self.agg(v_list.view(bs * nvars, pl, -1))
        v = v_list.view(bs, nvars, seq_len)
        
        v = v.permute(0, 2, 1)

        return v
