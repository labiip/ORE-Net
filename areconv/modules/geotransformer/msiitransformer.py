import pdb

import numpy as np
import torch
import torch.nn as nn

from areconv.modules.ops import pairwise_distance
from areconv.modules.transformer import SinusoidalPositionalEmbedding, MSIITransformer


class GeometricStructureEmbedding(nn.Module):
    def __init__(self, hidden_dim, sigma_d, sigma_a, angle_k, reduction_a='max'):
        super(GeometricStructureEmbedding, self).__init__()
        self.sigma_d = sigma_d
        self.sigma_a = sigma_a
        self.factor_a = 180.0 / (self.sigma_a * np.pi)
        self.angle_k = angle_k

        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.proj_d = nn.Linear(hidden_dim, 3)
        self.proj_a = nn.Linear(hidden_dim, 3)

        self.reduction_a = reduction_a
        if self.reduction_a not in ['max', 'mean']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction_a}.')

    @torch.no_grad()
    def get_embedding_indices(self, points):
        r"""Compute the indices of pair-wise distance embedding and triplet-wise angular embedding.

        Args:
            points: torch.Tensor (B, N, 3), input point cloud

        Returns:
            d_indices: torch.FloatTensor (B, N, N), distance embedding indices
            a_indices: torch.FloatTensor (B, N, N, k), angular embedding indices
        """
        batch_size, num_point, _ = points.shape

        dist_map = torch.sqrt(pairwise_distance(points, points))  # (B, N, N)
        d_indices = dist_map / self.sigma_d

        k = self.angle_k
        knn_indices = dist_map.topk(k=k + 1, dim=2, largest=False)[1][:, :, 1:]  # (B, N, k)
        knn_indices = knn_indices.unsqueeze(3).expand(batch_size, num_point, k, 3)  # (B, N, k, 3)
        expanded_points = points.unsqueeze(1).expand(batch_size, num_point, num_point, 3)  # (B, N, N, 3)
        knn_points = torch.gather(expanded_points, dim=2, index=knn_indices)  # (B, N, k, 3)
        ref_vectors = knn_points - points.unsqueeze(2)  # (B, N, k, 3)

        # ref_vectors = normals.unsqueeze(2)

        anc_vectors = points.unsqueeze(1) - points.unsqueeze(2)  # (B, N, N, 3)
        ref_vectors = ref_vectors.unsqueeze(2).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
        anc_vectors = anc_vectors.unsqueeze(3).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
        sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)  # (B, N, N, k)
        cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)  # (B, N, N, k)
        angles = torch.atan2(sin_values, cos_values)  # (B, N, N, k)
        a_indices = angles * self.factor_a

        return d_indices, a_indices

    def forward(self, points):
        d_indices, a_indices = self.get_embedding_indices(points)

        d_embeddings = self.embedding(d_indices)
        d_embeddings = self.proj_d(d_embeddings)

        a_embeddings = self.embedding(a_indices)
        a_embeddings = self.proj_a(a_embeddings)
        if self.reduction_a == 'max':
            a_embeddings = a_embeddings.max(dim=3)[0]
        else:
            a_embeddings = a_embeddings.mean(dim=3)
        # a_embeddings = a_embeddings[:, :, :, 0, :]
        embeddings = d_embeddings + a_embeddings
        # cat = torch.cat([d_indices[..., None], a_indices], dim=-1)

        return embeddings


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_heads,
        blocks,
        sigma_d,
        sigma_a,
        angle_k,
        dropout=None,
        activation_fn='ReLU',
        reduction_a='max',
    ):
        r"""MSII Transformer.

        Args:
            input_dim: input feature dimension
            output_dim: output feature dimension
            hidden_dim: hidden feature dimension
            num_heads: number of head in transformer
            blocks: list of 'self' or 'cross'
            sigma_d: temperature of distance
            sigma_a: temperature of angles
            angle_k: number of nearest neighbors for angular embedding
            activation_fn: activation function
            reduction_a: reduction mode of angular embedding ['max', 'mean']
        """
        super(Transformer, self).__init__()

        self.embedding = GeometricStructureEmbedding(hidden_dim, sigma_d, sigma_a, angle_k, reduction_a=reduction_a)

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.transformer = MSIITransformer(
            blocks, hidden_dim, num_heads, dropout=dropout, activation_fn=activation_fn, return_attention_scores=True, parallel=False
        )
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        ref_points,
        src_points,
        ref_feats,
        src_feats,
        ref_masks=None,
        src_masks=None,
    ):
        r"""MSII Transformer

        Args:
            ref_points (Tensor): (B, N, 3)
            src_points (Tensor): (B, M, 3)
            ref_feats (Tensor): (B, N, C)
            src_feats (Tensor): (B, M, C)
            ref_masks (Optional[BoolTensor]): (B, N)
            src_masks (Optional[BoolTensor]): (B, M)

        Returns:
            ref_feats: torch.Tensor (B, N, C)
            src_feats: torch.Tensor (B, M, C)
        """
        ref_embeddings_list = []
        src_embeddings_list = []
        for win_size in [2,4]:
            add_row_ref_num = int(np.ceil(ref_points.shape[1] / (win_size * win_size)) * win_size * win_size) - ref_points.shape[1]
            add_row_ref_4x4 = torch.cat([ref_points, torch.zeros((1, add_row_ref_num, 3), dtype=torch.float32).cuda()],1)
        
            ref_stage0 = self.embedding(
                add_row_ref_4x4.view(1, add_row_ref_4x4.shape[1] // (win_size*win_size), win_size*win_size, 3).view(add_row_ref_4x4.shape[1] //(win_size * win_size), (win_size * win_size), 3))

            add_row_src_num = int(np.ceil(src_points.shape[1] / (win_size * win_size)) * win_size * win_size) - src_points.shape[1]
            add_row_src_4x4 = torch.cat([src_points, torch.zeros((1, add_row_src_num, 3), dtype=torch.float32).cuda()], 1)
            src_stage0 = self.embedding(
                add_row_src_4x4.view(1, add_row_src_4x4.shape[1] // (win_size * win_size), (win_size * win_size), 3).view(add_row_src_4x4.shape[1] // (win_size * win_size), (win_size * win_size), 3))
            src_embeddings_list.append(src_stage0)   #[91, 4, 4, 4]  [23, 16, 16, 4]
            ref_embeddings_list.append(ref_stage0)   #[118, 4, 4, 4]  [30, 16, 16, 4]

        src_embeddings_list.append(self.embedding(src_points))  #[1, 363, 363, 4]
        ref_embeddings_list.append(self.embedding(ref_points))  #[1, 472, 472, 4]
        
        ref_feats = self.in_proj(ref_feats)   #(B, N, 256)
        src_feats = self.in_proj(src_feats)

        ref_feats, src_feats, scores_list = self.transformer(
            ref_feats,
            src_feats,
            ref_embeddings_list,
            src_embeddings_list,
            masks0=ref_masks,
            masks1=src_masks,
        )

        ref_feats = self.out_proj(ref_feats)
        src_feats = self.out_proj(src_feats)

        return ref_feats, src_feats, scores_list