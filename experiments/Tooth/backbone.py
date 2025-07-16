import copy
import pdb
import numpy as np

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from areconv.modules.layers import VNLinear, VNLinearLeakyReLU, VNLeakyReLU, VNStdFeature
from areconv.modules.ops import index_select

class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_model, d_k, d_v, h,dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout=nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model) == (B,N,C)
        :param keys: Keys (b_s, nk, d_model) == (B,N,C)
        :param values: Values (b_s, nk, d_model) == (B,N,C)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk); C=h*nk. True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class SimplifiedScaledDotProductAttention(nn.Module):

    def __init__(self, d_model, h,dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SimplifiedScaledDotProductAttention, self).__init__()

        self.d_model = d_model
        self.d_k = d_model//h
        self.d_v = d_model//h
        self.h = h

        self.fc_o = nn.Linear(h * self.d_v, d_model)
        self.dropout=nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = queries.view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = keys.view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = values.view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out

class PositionAttentionModule(nn.Module):

    def __init__(self,d_model, kernel_size):
        super().__init__()
        d_model = d_model*3
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size)
        self.pa=ScaledDotProductAttention(d_model,d_k=d_model,d_v=d_model,h=1)
    
    def forward(self,x):
        # (B, C, H, W)
        x = x.unsqueeze(0) #(B, N, C, 3)
        B, N, C, _= x.shape
        x = x.reshape(B, N, -1).transpose(1, 2)  #(B, C*3, N)
        y=self.conv1d(x) # (B, C*3, N) --> (B, C*3, N)
        y = y.permute(0,2,1) #(B, N, C*3)
        y=self.pa(y,y,y) #(B, N, C*3)
        return y

class ChannelAttentionModule(nn.Module):
    
    def __init__(self,d_model,kernel_size):
        super().__init__()
        d_model = d_model*3
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size)
        self.pa=SimplifiedScaledDotProductAttention(d_model, h=1)
    
    def forward(self,x):
        x = x.unsqueeze(0) #(B, N, C, 3)
        B, N, C, _= x.shape
        x = x.reshape(B, N, -1).transpose(1, 2)  #(B, C*3, N)
        y=self.conv1d(x)  # (B, C*3, N) --> (B, C*3, N)
        y = y.permute(0,2,1) #(B, N, C*3)
        y=self.pa(y,y,y) #(B, N, C*3)
        return y

class DAModule(nn.Module):

    def __init__(self,d_model,kernel_size):
        super().__init__()

        self.position_attention_module=PositionAttentionModule(d_model,kernel_size)
        self.channel_attention_module=ChannelAttentionModule(d_model,kernel_size)
        self.alpha = nn.Parameter(torch.tensor(0.5))  
        self.beta = nn.Parameter(torch.tensor(0.2))  
    
    def forward(self,input):
        N, C, _=input.shape
        p_out=self.position_attention_module(input) 
        c_out=self.channel_attention_module(input)  
        p_out=p_out.view(1,N,C,3) 
        c_out=c_out.view(1,N,C,3) 

        p_out = input + self.alpha * p_out.squeeze(0)
        c_out = input + self.beta * c_out.squeeze(0)

        return p_out+c_out

class CorrelationNet(nn.Module):

    def __init__(self, in_channel, out_channel, hidden_unit=[8, 8], last_bn=False, temp=1):
        super(CorrelationNet, self).__init__()
        self.vn_layer = VNLinearLeakyReLU(in_channel, out_channel * 2, dim=4, share_nonlinearity=False, negative_slope=0.2)
        self.hidden_unit = hidden_unit
        self.last_bn = last_bn
        self.mlp_convs_hidden = nn.ModuleList()
        self.mlp_bns_hidden = nn.ModuleList()
        self.temp = temp

        hidden_unit = list() if hidden_unit is None else copy.deepcopy(hidden_unit)
        hidden_unit.insert(0, out_channel * 2)
        hidden_unit.append(out_channel)
        for i in range(1, len(hidden_unit)):  
            self.mlp_convs_hidden.append(nn.Conv1d(hidden_unit[i - 1], hidden_unit[i], 1,
                                                   bias=False if i < len(hidden_unit) - 1 else not last_bn))
            if i < len(hidden_unit) - 1 or last_bn:
                self.mlp_bns_hidden.append(nn.BatchNorm1d(hidden_unit[i]))

    def forward(self, xyz):
        # xyz : N * D * 3 * k
        N, _, _, K = xyz.size()
        scores = self.vn_layer(xyz)
        scores = torch.norm(scores, p=2, dim=2)  
        for i, conv in enumerate(self.mlp_convs_hidden):
            if i < len(self.mlp_convs_hidden) - 1:
                scores = F.relu(self.mlp_bns_hidden[i](conv(scores)))
            else:  # if the output layer, no ReLU
                scores = conv(scores)
                if self.last_bn:
                    scores = self.mlp_bns_hidden[i](scores)
        scores = F.softmax(scores/self.temp, dim=1)
        return scores

class ARE_Conv_Block(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, share_nonlinearity=False):
        super(ARE_Conv_Block, self).__init__()
        self.kernel_size = kernel_size
        self.score_net = CorrelationNet(in_channel=3, out_channel=self.kernel_size, hidden_unit=[self.kernel_size])

        in_dim = in_dim + 2   # 1 + 2: [xyz, mean, cross]
        tensor1 = nn.init.kaiming_normal_(torch.empty(self.kernel_size, in_dim, out_dim // 2)).contiguous()
        tensor1 = tensor1.permute(1, 0, 2).reshape(in_dim, self.kernel_size * out_dim // 2)
        self.weightbank = nn.Parameter(tensor1, requires_grad=True)

        self.relu = VNLeakyReLU(out_dim//2, share_nonlinearity)
        self.unary = VNLinearLeakyReLU(out_dim//2, out_dim)

    def forward(self, q_pts, s_pts, s_feats, neighbor_indices):  # points_list[1], points_list[0], feats_s1, subsampling_list[0]
        """
        q_pts N1 * 3
        s_pts N2 * 3
        q_feats N1 * D * 3
        s_feats N2 * D * 3
        neighbor_indices   N1 * k
        """
        N, K = neighbor_indices.shape

        # compute relative coordinates
        pts = (s_pts[neighbor_indices] - q_pts[:, None]).unsqueeze(1).permute(0, 1, 3, 2)  # [N, 1, K, 3] --> [N, 1, 3, K]  
        centers = pts.mean(-1, keepdim=True).repeat(1, 1, 1, K)
        cross = torch.cross(pts, centers, dim=2)
        local_feats = torch.cat([pts, centers, cross], 1) # [N, 3, 3, K] rotation equivariant spatial features

        # predict correlation scores
        scores = self.score_net(local_feats) # [N, kernel_size,  K]

        # use correlation scores to assemble features
        pro_feats = torch.einsum('ncdk,cf->nfdk', local_feats, self.weightbank)
        pro_feats = pro_feats.reshape(N,  self.kernel_size, -1, 3, K)
        pro_feats = (pro_feats * scores[:, :, None, None]).sum(1) # [N, D/2, 3, K]

        # use L2 Norm instead of VNBatchNorm to reduce computation cost and accelerate convergence
        normed_feats = F.normalize(pro_feats, p=2, dim=2)
        # mean pooling
        new_feats = normed_feats.mean(-1)
        # applying VN ReLU after pooling to reduce computation cost
        new_feats = self.relu(new_feats)
        # mapping D/2 -> D
        new_feats = self.unary(new_feats)  # [N, D, 3]

        return new_feats

class ARE_Conv_Resblock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, shortcut_linear=False, share_nonlinearity=False, conv_info=None):
        super(ARE_Conv_Resblock, self).__init__()
        self.kernel_size = kernel_size
        self.score_net = CorrelationNet(in_channel=3, out_channel=self.kernel_size, hidden_unit=[self.kernel_size])

        self.conv_way = conv_info["conv_way"]
        self.use_xyz = conv_info["use_xyz"]
        conv_dim = in_dim * 2 if self.conv_way == 'edge_conv' else in_dim
        if self.use_xyz: conv_dim += 1
        tensor1 = nn.init.kaiming_normal_(torch.empty(self.kernel_size, conv_dim, out_dim//2)).contiguous()
        tensor1 = tensor1.permute(1, 0, 2).reshape(conv_dim, self.kernel_size * out_dim//2)
        self.weightbank = nn.Parameter(tensor1, requires_grad=True)

        self.relu = VNLeakyReLU(out_dim//2, share_nonlinearity)
        self.shortcut_proj = VNLinear(in_dim, out_dim) if shortcut_linear else nn.Identity()
        self.unary = VNLinearLeakyReLU(out_dim//2, out_dim)
    def forward(self, q_pts, s_pts, s_feats, neighbor_indices):
        """
        q_pts N1 * 3
        s_pts N2 * 3
        q_feats N1 * D * 3
        s_feats N2 * D * 3
        neighbor_indices   N1 * k
        """

        N, K = neighbor_indices.shape
        pts = (s_pts[neighbor_indices] - q_pts[:, None]).unsqueeze(1).permute(0, 1, 3, 2)    # N1 *1 * 3 * k
        # compute relative coordinates
        center = pts.mean(-1, keepdim=True).repeat(1, 1, 1, K)
        cross = torch.cross(pts, center, dim=2)
        local_feats = torch.cat([pts, center, cross], 1)# [N, 3, 3, K] rotation equivariant spatial features
        # predict correlation scores
        scores = self.score_net(local_feats)
        # gather neighbors features
        neighbor_feats = s_feats[neighbor_indices, :].permute(0, 2, 3, 1)                            # N1  D * 3 k
        # shortcut
        identify = neighbor_feats[..., 0]
        identify = self.shortcut_proj(identify)
        # get edge features
        if self.conv_way == 'edge_conv':
            q_feats = neighbor_feats[..., 0:1]
            neighbor_feats = torch.cat([neighbor_feats - q_feats, neighbor_feats], 1)
        # use relative coordinates
        if self.use_xyz:
            neighbor_feats = torch.cat([neighbor_feats, pts], 1)
        # use correlation scores to assemble features
        pro_feats = torch.einsum('ncdk,cf->nfdk', neighbor_feats, self.weightbank)
        pro_feats = pro_feats.reshape(N, self.kernel_size, -1, 3, K)
        pro_feats = (pro_feats * scores[:, :, None, None]).sum(1)

        # use L2 Norm instead of VNBatchNorm to reduce computation cost and accelerate convergence
        normed_feats = F.normalize(pro_feats, p=2, dim=2)

        # mean pooling
        new_feats = normed_feats.mean(-1)
        # apply VN ReLU after pooling to reduce computation cost
        new_feats = self.relu(new_feats)

        # map D/2 -> D
        new_feats = self.unary(new_feats)  # [N, D, 3]
        # add shortcut
        new_feats = new_feats + identify

        return new_feats

class AREConvFPN(nn.Module):
    def __init__(self, init_dim, output_dim, kernel_size, share_nonlinearity=False, conv_way='edge_conv', use_xyz=True, use_encoder_re_feats=True):
        super(AREConvFPN, self).__init__()
        conv_info = {'conv_way': conv_way, 'use_xyz': use_xyz}
        self.use_encoder_re_feats = use_encoder_re_feats
        self.encoder2_1 = ARE_Conv_Block(1, init_dim // 3, kernel_size, share_nonlinearity=share_nonlinearity)
        self.encoder2_2 = ARE_Conv_Resblock(init_dim // 3, 2 * init_dim // 3, kernel_size, shortcut_linear=True, share_nonlinearity=share_nonlinearity, conv_info=conv_info)
        self.encoder2_3 = ARE_Conv_Resblock(2 * init_dim // 3, 2 * init_dim // 3, kernel_size, shortcut_linear=False, share_nonlinearity=share_nonlinearity, conv_info=conv_info)
        self.da2_1 = DAModule(d_model=64,kernel_size=1)
        self.da2_2 = DAModule(d_model=64,kernel_size=1)

        self.encoder3_1 = ARE_Conv_Resblock(2 * init_dim // 3, 4 * init_dim // 3, kernel_size, shortcut_linear=True, share_nonlinearity=share_nonlinearity, conv_info=conv_info)
        self.encoder3_2 = ARE_Conv_Resblock(4 * init_dim // 3, 4 * init_dim // 3, kernel_size, shortcut_linear=False, share_nonlinearity=share_nonlinearity, conv_info=conv_info)
        self.encoder3_3 = ARE_Conv_Resblock(4 * init_dim // 3, 4 * init_dim // 3, kernel_size, shortcut_linear=False, share_nonlinearity=share_nonlinearity, conv_info=conv_info)
        self.da3_1 = DAModule(d_model=128,kernel_size=1)
        self.da3_2 = DAModule(d_model=128,kernel_size=1)

        self.encoder4_1 = ARE_Conv_Resblock(4 * init_dim // 3, 8 * init_dim // 3, kernel_size, shortcut_linear=True, share_nonlinearity=share_nonlinearity, conv_info=conv_info)
        self.encoder4_2 = ARE_Conv_Resblock(8 * init_dim // 3, 8 * init_dim // 3, kernel_size, shortcut_linear=False, share_nonlinearity=share_nonlinearity, conv_info=conv_info)
        self.encoder4_3 = ARE_Conv_Resblock(8 * init_dim // 3, 8 * init_dim // 3, kernel_size, shortcut_linear=False, share_nonlinearity=share_nonlinearity, conv_info=conv_info)
        self.da4_1 = DAModule(d_model=256,kernel_size=1)
        self.da4_2 = DAModule(d_model=256,kernel_size=1)

        self.coarse_RI_head = VNLinear(8 * init_dim // 3, 8 * init_dim // 3)
        self.coarse_std_feature = VNStdFeature(8 * init_dim // 3, dim=3, normalize_frame=True, share_nonlinearity=share_nonlinearity)

        self.decoder3 = VNLinearLeakyReLU(12 * init_dim // 3, 4 * init_dim // 3, dim=3, share_nonlinearity=share_nonlinearity)
        self.decoder2 = VNLinearLeakyReLU(6 * init_dim // 3, output_dim // 3, dim=3, share_nonlinearity=share_nonlinearity)
        self.RI_head = VNLinear(output_dim // 3, output_dim // 3)
        self.RE_head = VNLinear(output_dim // 3, output_dim // 3)

        self.fine_std_feature = VNStdFeature(output_dim // 3, dim=3, normalize_frame=True, share_nonlinearity=share_nonlinearity)

        self.matching_score_proj = nn.Linear(output_dim // 3 * 3, 1)

    def forward(self, data_dict):

        points_list = data_dict['points']
        neighbors_list = data_dict['neighbors']
        subsampling_list = data_dict['subsampling']
        upsampling_list = data_dict['upsampling']
        ref_length_f = data_dict['lengths'][1][0]
        ref_length_s = data_dict['lengths'][2][0]
        ref_length_c = data_dict['lengths'][-1][0]
        feats_s1 = points_list[0][:, None]

        feats_s2 = self.encoder2_1(points_list[1], points_list[0], feats_s1, subsampling_list[0])
        feats_s2 = self.encoder2_2(points_list[1], points_list[1], feats_s2, neighbors_list[1])
        feats_s2 = self.encoder2_3(points_list[1], points_list[1], feats_s2, neighbors_list[1])

        ref_feats_s2 = feats_s2[:ref_length_f]
        src_feats_s2 = feats_s2[ref_length_f:]
        ref_feats_s2 = self.da2_1(ref_feats_s2)
        src_feats_s2 = self.da2_2(src_feats_s2)
        feats_s2 = torch.cat([ref_feats_s2, src_feats_s2], dim=0)

        feats_s3 = self.encoder3_1(points_list[2], points_list[1], feats_s2, subsampling_list[1])
        feats_s3 = self.encoder3_2(points_list[2], points_list[2], feats_s3, neighbors_list[2])
        feats_s3 = self.encoder3_3(points_list[2], points_list[2], feats_s3, neighbors_list[2])

        ref_feats_s3 = feats_s3[:ref_length_s]
        src_feats_s3 = feats_s3[ref_length_s:]
        ref_feats_s3 = self.da3_1(ref_feats_s3)
        src_feats_s3 = self.da3_2(src_feats_s3)
        feats_s3 = torch.cat([ref_feats_s3, src_feats_s3], dim=0)

        feats_s4 = self.encoder4_1(points_list[3], points_list[2], feats_s3, subsampling_list[2])
        feats_s4 = self.encoder4_2(points_list[3], points_list[3], feats_s4, neighbors_list[3])
        feats_s4 = self.encoder4_3(points_list[3], points_list[3], feats_s4, neighbors_list[3])

        ref_feats_s4 = feats_s4[:ref_length_c]
        src_feats_s4 = feats_s4[ref_length_c:]
        ref_feats_s4 = self.da4_1(ref_feats_s4)
        src_feats_s4 = self.da4_2(src_feats_s4)
        feats_s4 = torch.cat([ref_feats_s4, src_feats_s4], dim=0)

        coarse_feats = self.coarse_RI_head(feats_s4)
        ri_feats_c, _ = self.coarse_std_feature(coarse_feats)

        ri_feats_c = ri_feats_c.reshape(ri_feats_c.shape[0], -1)

        up1 = upsampling_list[1]
        latent_s3 = index_select(feats_s4, up1[:, 0], dim=0)
        latent_s3 = torch.cat([latent_s3, feats_s3], dim=1)
        latent_s3 = self.decoder3(latent_s3)   

        up2 = upsampling_list[0]
        latent_s2 = index_select(latent_s3, up2[:, 0], dim=0)
        latent_s2 = torch.cat([latent_s2, feats_s2], dim=1)  
        latent_s2 = self.decoder2(latent_s2)   

        ri_feats = self.RI_head(latent_s2)
        re_feats = self.RE_head(latent_s2)

        ri_feats_f, local_rot = self.fine_std_feature(ri_feats)
        ri_feats_f = ri_feats_f.reshape(ri_feats_f.shape[0], -1)
        m_scores = self.matching_score_proj(ri_feats_f).sigmoid().squeeze()
        if not self.training and self.use_encoder_re_feats:
            # using rotation equivariant features from encoder to solve transformation may generate better hypotheses,
            # probably because a larger receptive field would contaminate rotation equivariant features
            re_feats_f = feats_s2
        else:
            re_feats_f = re_feats
        return re_feats_f, ri_feats_f, feats_s4, ri_feats_c, m_scores