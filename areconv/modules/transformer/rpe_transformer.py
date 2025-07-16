r"""Transformer with Relative Positional Embeddings.

Relative positional embedding is further projected in each multi-head attention layer.

The shape of input tensor should be (B, N, C). Implemented with `nn.Linear` and `nn.LayerNorm` (with affine).
"""
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from IPython import embed
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from areconv.modules.layers import build_dropout_layer
from areconv.modules.transformer.output_layer import AttentionOutput

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MSMLMultiHeadAttention(nn.Module):
    def __init__(self, d_model, stage, dropout=None):
        super(MSMLMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_k = nn.Linear(d_model, d_model)
        self.proj_p = nn.Linear(3, 1)  # 位置编码投影
        self.dropout = build_dropout_layer(dropout)
        self.stage = stage
        self.elu = nn.ELU()

    def forward(self, input, embed_qk, key_weights=None, key_masks=None, attention_factors=None):
        # --- 1. 窗口划分 --
        if self.stage == 0:
            win_w, win_h = 2, 2
        elif self.stage == 1:
            win_w, win_h = 4, 4
        elif self.stage == 2:
            win_w, win_h = 4, 4

        B, N, C = input.shape
        add_row_num = int(np.ceil(N / (win_w * win_h)) * win_w * win_h) - N

        # --- 2. 投影查询、键、值 ---
        q = self.elu(self.proj_q(input)) + 1.0  # (B, N, C)
        k = self.elu(self.proj_k(input)) + 1.0
        v = input

        # 填充零以适应窗口划分
        q_pad = torch.cat([q, torch.zeros((B, add_row_num, C), device=input.device)], dim=1)
        k_pad = torch.cat([k, torch.zeros((B, add_row_num, C), device=input.device)], dim=1)
        v_pad = torch.cat([v, torch.zeros((B, add_row_num, C), device=input.device)], dim=1)

        # --- 3. 窗口重塑 ---
        q_win = q_pad.view(B, -1, win_w * win_h, C)  # (B, num_win, win_size, C)
        k_win = k_pad.view(B, -1, win_w * win_h, C)
        v_win = v_pad.view(B, -1, win_w * win_h, C)
        if self.stage == 2:
        # --- 4. 位置编码处理 ---
            proj_p = self.proj_p(embed_qk[self.stage - 1])  # (B, win_num, win_size, 1)
        else:
            proj_p = self.proj_p(embed_qk[self.stage])  # (B, win_num, win_size, 1)
        p = proj_p.squeeze(-1)  # (B, win_num, win_size)

        # --- 5. 线性注意力计算 ---
        # 特征映射后的查询和键
        phi_q = self.elu(q_win) + 1.0  # (B, num_win, win_size, C)
        phi_k = self.elu(k_win) + 1.0

        # 线性注意力核心计算
        KV = torch.einsum('bwmc,bwmd->bmcd', phi_k, v_win)  # (B, num_win, C, C)
        attention = torch.einsum('bwnc,bmcd->bwnmd', phi_q, KV)  # (B, num_win, win_size, C)

        # 添加位置偏置
        attention = attention + p.unsqueeze(-1)  # (B, num_win, win_size, C)

        # --- 6. 窗口合并 ---
        output = attention.view(B, -1, C)  # (B, total_tokens, C)
        output = output[:, :N, :]  # 移除填充部分

        return output, None  # 返回输出和空注意力分数（线性注意力无需保存分数）

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def create_mask(H,W,win_h,win_w,window_size,shift_size):
    Hp = int(np.ceil(H / win_h)) * window_size
    Wp = int(np.ceil(W / win_w)) * window_size
    img_mask = torch.zeros((1, Hp, Wp, 1)).cuda()  # 1 Hp Wp 1
    h_slices = (slice(0, -1*window_size),
                slice(-1*window_size, -1*shift_size),
                slice(-1*shift_size, None))
    w_slices = (slice(0, -1*window_size),
                slice(-1*window_size, -1*shift_size),
                slice(-1*shift_size, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask

class SWIFT_MSMLMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, stage,dropout=None):
        super(SWIFT_MSMLMultiHeadAttention, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError('`d_model` ({}) must be a multiple of `num_heads` ({}).'.format(d_model, num_heads))

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads

        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)
        # self.proj_p = nn.Linear(self.d_model, self.d_model)
        self.proj_p = nn.Linear(3, 1)

        self.dropout = build_dropout_layer(dropout)
        self.stage = stage

    def forward(self, input_q, input_k, input_v, embed_qk, key_weights=None, key_masks=None, attention_factors=None):
        r"""Scaled Dot-Product Attention with Pre-computed Relative Positional Embedding (forward)

        Args:
            input_q: torch.Tensor (B, N, C)
            input_k: torch.Tensor (B, M, C)
            input_v: torch.Tensor (B, M, C)
            embed_qk: torch.Tensor (B, N, M, C), relative positional embedding
            key_weights: torch.Tensor (B, M), soft masks for the keys
            key_masks: torch.Tensor (B, M), True if ignored, False if preserved
            attention_factors: torch.Tensor (B, N, M)

        Returns:
            hidden_states: torch.Tensor (B, C, N)
            attention_scores: torch.Tensor (B, H, N, M)
        """
        windows_attent_filg = False

        if(self.stage==0):
            win_w = 2
            win_h = 2
            mask = torch.zeros((win_w*win_h,win_w*win_h)).cuda()

            mask_len = win_w*win_h//2

            mask[0:mask_len,mask_len:] = -100
            mask[mask_len:,0:mask_len] = -100
            windows_attent_filg = True
        elif(self.stage==1):
            win_w = 4
            win_h = 4

            mask_len = win_w * win_h // 2

            mask = torch.zeros((win_w*win_h, win_w*win_h)).cuda()
            mask[0:mask_len,mask_len:] = -100
            mask[mask_len:,0:mask_len] = -100
            windows_attent_filg = True

        elif(self.stage==2):
            win_w = 4
            win_h = 4

            mask_len = win_w * win_h // 2

            mask = torch.zeros((win_w*win_h, win_w*win_h)).cuda()
            mask[0:mask_len,mask_len:] = -100
            mask[mask_len:,0:mask_len] = -100
            windows_attent_filg = True

        if(windows_attent_filg == True):

            add_row_num = int(np.ceil(input_q.shape[1] / (win_w * win_h)) * win_w * win_h) - input_q.shape[1]

            input_q_cat = torch.cat([input_q, torch.zeros((1, add_row_num, self.d_model), dtype=torch.float32).cuda()], 1)
            input_q_cat_roll = torch.zeros((input_q_cat.shape[0],input_q_cat.shape[1],input_q_cat.shape[2]), dtype=torch.float32).cuda()
            input_q_cat_roll[:,0:input_q_cat.shape[1]-(win_w * win_h)//2,:]  = input_q_cat[:,((win_w * win_h)//2):,:]
            input_q_cat_roll[:,input_q_cat.shape[1] - (win_w * win_h)//2:,:] = input_q_cat[:,0:(win_w * win_h)//2,:]

            q = rearrange(self.proj_q(input_q_cat_roll)
                          , 'b n (h c) -> b h n c', h=self.num_heads)
            k = rearrange(self.proj_k(input_q_cat_roll)
                          , 'b m (h c) -> b h m c', h=self.num_heads)
            v = rearrange(self.proj_k(input_q_cat_roll)
                          , 'b m (h c) -> b h m c', h=self.num_heads)
            if self.stage == 2:
                proj = self.proj_p(embed_qk[self.stage-1])
            else:
                proj = self.proj_p(embed_qk[self.stage])
            p = proj.repeat(1, 1, 1, self.num_heads).permute(0, 3, 1, 2)

         
            q = q.view(q.shape[0], self.num_heads, q.shape[2] // (win_w*win_h),(win_w*win_h), q.shape[3]).transpose(1, 2)
            q = q.view(q.shape[0]*q.shape[1],self.num_heads,q.shape[-2],q.shape[-1])

            k = k.view(k.shape[0], self.num_heads, k.shape[2] // (win_w*win_h),(win_w*win_h), k.shape[3]).transpose(1, 2)
            k = k.view(k.shape[0]*k.shape[1],self.num_heads,k.shape[-2],k.shape[-1])

            v = v.view(v.shape[0], self.num_heads, v.shape[2] // (win_w*win_h), (win_w * win_h), v.shape[3]).transpose(1, 2)
            v = v.view(v.shape[0] * v.shape[1], self.num_heads, v.shape[-2], v.shape[-1])

            attention_scores_e = torch.einsum('bhnc,bhmc->bhnm', q, k)
            attention_scores = (attention_scores_e + p) / self.d_model_per_head ** 0.5

            if attention_factors is not None:
                attention_scores = attention_factors.unsqueeze(1) * attention_scores
            if key_weights is not None:
                attention_scores = attention_scores * key_weights.unsqueeze(1).unsqueeze(1)
            if key_masks is not None:
                attention_scores = attention_scores.masked_fill(key_masks.unsqueeze(1).unsqueeze(1), float('-inf'))
            attention_scores = attention_scores + mask
            attention_scores = F.softmax(attention_scores, dim=-1)
            attention_scores = self.dropout(attention_scores)

            hidden_states = torch.matmul(attention_scores, v)
            first = hidden_states.shape[0]
            second = hidden_states.shape[1]
            third = hidden_states.shape[2]
            fourth = hidden_states.shape[3]


            hidden_states = hidden_states.view(first * second * third, fourth).view(1, second, first * third, fourth)
            hidden_states = rearrange(hidden_states, 'b h n c -> b n (h c)')

            hidden_states_reverse = torch.zeros((hidden_states.shape[0], hidden_states.shape[1], hidden_states.shape[2]),
                                           dtype=torch.float32).cuda()

            hidden_states_reverse[:, 0:(win_w * win_h) // 2, :] = hidden_states[:,hidden_states.shape[1]-((win_w * win_h) // 2):, :]
            hidden_states_reverse[:,(win_w * win_h) // 2:,:] = hidden_states[:,0:hidden_states.shape[1]-((win_w * win_h) // 2),:]

            return hidden_states[:, 0:hidden_states.shape[1] - add_row_num, :], attention_scores

        else:
            q = rearrange(self.proj_q(input_q), 'b n (h c) -> b h n c', h=self.num_heads)
            k = rearrange(self.proj_k(input_k), 'b m (h c) -> b h m c', h=self.num_heads)
            v = rearrange(self.proj_v(input_v), 'b m (h c) -> b h m c', h=self.num_heads)

            proj = self.proj_p(embed_qk[self.stage])
            p = proj.repeat(1, 1, 1, self.num_heads).permute(0, 3, 1, 2)

            attention_scores_e = torch.einsum('bhnc,bhmc->bhnm', q, k)
            attention_scores = (attention_scores_e + p) / self.d_model_per_head ** 0.5
            if attention_factors is not None:
                attention_scores = attention_factors.unsqueeze(1) * attention_scores
            if key_weights is not None:
                attention_scores = attention_scores * key_weights.unsqueeze(1).unsqueeze(1)
            if key_masks is not None:
                attention_scores = attention_scores.masked_fill(key_masks.unsqueeze(1).unsqueeze(1), float('-inf'))
            attention_scores = F.softmax(attention_scores, dim=-1)
            attention_scores = self.dropout(attention_scores)

            hidden_states = torch.matmul(attention_scores, v)

            hidden_states = rearrange(hidden_states, 'b h n c -> b n (h c)')
            return hidden_states, attention_scores


class MSMAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, stage, dropout=None):
        super(MSMAttentionLayer, self).__init__()
        self.attention = MSMLMultiHeadAttention(d_model, stage, dropout=dropout)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = build_dropout_layer(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.act_proj = nn.Linear(d_model, d_model)
        self.act = nn.SiLU()
        self.proj = nn.LayerNorm(d_model)
        self.conv = nn.Conv1d(d_model, d_model, 1)

    def forward(
        self,
        input_states,
        position_states,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
    ):
        act_res = self.act(self.act_proj(input_states))
        input_states = self.proj(input_states)  
        input_states = input_states.permute(0, 2, 1)  
        input_states = self.conv(input_states)  
        input_states = input_states.permute(0, 2, 1)
        input_states = self.act(input_states)
        hidden_states, attention_scores = self.attention(
            input_states,
            position_states,
            key_weights=memory_weights,
            key_masks=memory_masks,
            attention_factors=attention_factors,
        )
        hidden_states = self.linear(hidden_states * act_res)
        hidden_states = self.dropout(hidden_states)
        output_states = self.norm(hidden_states + input_states)
        return output_states, attention_scores


class swift_MSMAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads,stage, dropout=None):
        super(swift_MSMAttentionLayer, self).__init__()
        self.swift_attention = SWIFT_MSMLMultiHeadAttention(d_model, num_heads,stage, dropout=dropout)
        self.swift_linear = nn.Linear(d_model, d_model)
        self.dropout = build_dropout_layer(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_states,
        memory_states,
        position_states,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
    ):
        hidden_states, attention_scores = self.swift_attention(
            input_states,
            memory_states,
            memory_states,
            position_states,
            key_weights=memory_weights,
            key_masks=memory_masks,
            attention_factors=attention_factors,
        )
        hidden_states = self.swift_linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output_states = self.norm(hidden_states + input_states)
        return output_states, attention_scores


class MSMLTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, stage, dropout=None, activation_fn='ReLU', drop_path = 0.):
        super(MSMLTransformerLayer, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attention = MSMAttentionLayer(d_model, num_heads, stage, dropout=dropout)
        self.swift_attention = swift_MSMAttentionLayer(d_model, num_heads, stage, dropout=dropout)
        self.output = AttentionOutput(d_model, dropout=dropout, activation_fn=activation_fn)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = Mlp(in_features=d_model, hidden_features=int(d_model * 4), act_layer=nn.GELU, drop=0.)

    def forward(
        self,
        input_states,
        position_states,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
    ):
        shortcut = input_states
        input_states = self.norm(input_states)
        hidden_states, attention_scores = self.attention(
            input_states,
            position_states,
            memory_weights=memory_weights,
            memory_masks=memory_masks,
            attention_factors=attention_factors,
        )
        hidden_states = shortcut + self.drop_path(hidden_states)
        hidden_states = hidden_states + self.drop_path(self.mlp(self.norm2(hidden_states)))

        hidden_states, attention_scores = self.swift_attention(
            hidden_states,
            hidden_states,
            position_states,
            memory_weights=memory_weights,
            memory_masks=memory_masks,
            attention_factors=attention_factors,
        )

        output_states = self.output(hidden_states)
        return output_states, attention_scores