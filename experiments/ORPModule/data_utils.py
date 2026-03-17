import os, sys, glob, h5py
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import minkowski
import open3d as o3d
import pickle
from scipy.spatial import cKDTree
from typing import Optional
from sklearn.neighbors import NearestNeighbors
import numpy as np

def farthest_point_sampling(points, num_samples):
    """统一数据类型处理"""
    # 确保输入为Tensor且在CPU
    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points).float()  # 转换为CPU Tensor
    
    N, _ = points.shape
    centroids = torch.zeros(num_samples, dtype=torch.long)
    distance = torch.ones(N) * 1e10
    
    farthest = torch.randint(0, N, (1,)).item()
    
    for i in range(num_samples):
        centroids[i] = farthest
        centroid = points[farthest].unsqueeze(0)  # 保持CPU计算
        
        dist = torch.sum((points - centroid)**2, dim=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.argmax(distance)
    
    return centroids.numpy()  # 返回numpy数组以兼容后续处理

def farthest_point_sampling_gpu(points, num_samples):
    # 确保输入为Tensor且转移到GPU
    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points).float().cuda()
    else:
        points = points.float().cuda()
    
    N, _ = points.shape
    device = points.device
    centroids = torch.zeros(num_samples, dtype=torch.long, device=device) 
    distance = torch.ones(N, device=device) * 1e10
    
    farthest = torch.randint(0, N, (1,), device=device).squeeze()
    
    for i in range(num_samples):
        centroids[i] = farthest
        centroid = points[farthest].unsqueeze(0)
        
        dist = torch.sum((points - centroid)**2, dim=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.argmax(distance)
    
    return centroids.cpu().numpy()

def apply_transform(points: np.ndarray, transform: np.ndarray, normals: Optional[np.ndarray] = None):
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    points = np.matmul(points, rotation.T) + translation
    if normals is not None:
        normals = np.matmul(normals, rotation.T)
        return points, normals
    else:
        return points

def get_nearest_neighbor(
    q_points: np.ndarray,
    s_points: np.ndarray,
    return_index: bool = False,
):
    r"""Compute the nearest neighbor for the query points in support points."""
    s_tree = cKDTree(s_points)
    distances, indices = s_tree.query(q_points, k=1)
    if return_index:
        return distances, indices
    else:
        return distances
    
def compute_overlap_mask(ref_points, src_points, transform=None, positive_radius=0.05):
    r"""Compute the overlap of two point clouds."""
    if transform is not None:
        src_points = apply_transform(src_points, transform)
    ref_nn_distances = get_nearest_neighbor(ref_points, src_points)
    ref_overlap_mask = ref_nn_distances < positive_radius
    src_nn_distances = get_nearest_neighbor(src_points, ref_points)
    src_overlap_mask = src_nn_distances < positive_radius
    return ref_overlap_mask, src_overlap_mask

def compute_overlap(ref_points, src_points, transform=None, positive_radius=0.1):
    r"""Compute the overlap of two point clouds."""
    if transform is not None:
        src_points = apply_transform(src_points, transform)
    nn_distances = get_nearest_neighbor(ref_points, src_points)
    overlap = np.mean(nn_distances < positive_radius)
    return overlap

def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

class Tooth(Dataset):
    def __init__(self, subset='train', partial_overlap=2):
        self.partial_overlap = partial_overlap  # 部分重叠的点云个数：0，1，2
        self.subset = subset
        self.dataset_root = '/home/kaiyue.bi/00Data/Tooth_Registration_Dataset/'

        if subset.lower() == "train":
            pkl_dir = osp.join(self.dataset_root, "train")
        else:
            pkl_dir = osp.join(self.dataset_root, "test")

        self.pkl_files = sorted(glob.glob(osp.join(pkl_dir, "*.pkl")))
        if len(self.pkl_files) == 0:
            raise RuntimeError(f"在 {pkl_dir} 中没有找到 pkl 文件！")

    def __getitem__(self, item):
        data_dict = {}

        pkl_file = self.pkl_files[item]
        data_dict['name'] = os.path.basename(pkl_file)

        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        src_points_raw = data['src_pcd'] 
        ref_points_raw = data['tgt_pcd']  

        indices = farthest_point_sampling(src_points_raw, 5000)
        src_points = src_points_raw[indices]
        indices = farthest_point_sampling(ref_points_raw, 5000)
        ref_points = ref_points_raw[indices]

        transform = data['gt_pose']     # 4x4 numpy 数组
        rotation = transform[:3, :3] 
        translation = transform[:3, 3]

        ref_overlap_mask, src_overlap_mask = compute_overlap_mask(ref_points, src_points, transform, 0.05)

        # 部分重叠
    
        gt_mask_src = torch.Tensor(src_overlap_mask)
        gt_mask_tgt = torch.Tensor(ref_overlap_mask)

            # 打乱点的顺序
        state = np.random.get_state()
        src_points = np.random.permutation(src_points).T
        np.random.set_state(state)
        gt_mask_src = np.random.permutation(gt_mask_src).T

        return src_points.astype('float32'), ref_points.astype('float32'), rotation.astype('float32'), \
                   translation.astype('float32'), transform.astype('float32'), gt_mask_src, gt_mask_tgt, data_dict['name']


    def __len__(self):
        return len(self.pkl_files)