import torch
import numpy as np
from model import OverlapNet
from data_utils import Tooth
from torch.utils.data import DataLoader
import open3d as o3d
import os
import pickle 

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def save_overlap_cloud(points, mask, filename):
    """
    保存重叠部分点云
    :param points: 原始点云 (N, 3)
    :param mask: 预测mask (N,)
    :param filename: 输出文件名
    """
    # 转换数据类型和形状
    points = np.asarray(points, dtype=np.float64)
    
    # 确保形状为(N, 3)
    if points.shape[1] != 3:
        points = points.T
    
    # 筛选重叠点
    overlap_points = points[mask]
    
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(overlap_points)
    
    # 保存点云
    o3d.io.write_point_cloud(filename, pcd)

def test():
    setup_seed(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建输出目录
    output_dir = "./overlap_clouds_train_pkl"
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化模型
    net = OverlapNet(src_overlap=3000, tgt_overlap=2000, all_points=5000).to(device)
    
    # 加载训练好的模型
    checkpoint = torch.load("./checkpoint/ckpt_best400.80.8.pth", map_location=device)
    net.load_state_dict(checkpoint['net'])
    net.eval()

    test_dataset = Tooth(subset='train', partial_overlap=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for idx, (src_pts, tgt_pts, _, _, transform, gt_mask_src, gt_mask_tgt, names) in enumerate(test_loader):
            # 数据预处理
            src = src_pts.to(device)                      # [1, 3, N]
            tgt = tgt_pts.permute(0, 2, 1).to(device)      # [1, N, 3]
        
            # 模型预测
            mask_src, mask_tgt, mask_src_idx, mask_tgt_idx = net(src, tgt)
        
            # 转换数据格式
            src_pts_np = src_pts.squeeze(0).numpy().T      # [N, 3]
            tgt_pts_np = tgt_pts.permute(0, 2, 1).squeeze(0).numpy().T      # [N, 3]
        
            # 获取预测mask
            pred_src_mask = mask_src_idx.squeeze().cpu().numpy().astype(bool)
            pred_tgt_mask = mask_tgt_idx.squeeze().cpu().numpy().astype(bool)
        
            # 提取重叠点云
            src_overlap = src_pts_np[pred_src_mask]  # [N_overlap, 3]
            tgt_overlap = tgt_pts_np[pred_tgt_mask]  # [M_overlap, 3]
        
            # 获取GT变换矩阵
            transform_gt = transform.squeeze(0).cpu().numpy()  # [4, 4]
        
            # 构建数据字典
            data_dict = {
                'src_pcd': src_overlap,
                'tgt_pcd': tgt_overlap,
                'gt_pose': transform_gt
            }
            for name in names:
                clean_name = name.replace("(", "").replace(")", "").replace("'", "").replace(",", "").strip()

            # 保存为pkl文件
            output_path = os.path.join(output_dir, f"{clean_name}")
            with open(output_path, 'wb') as f:
                pickle.dump(data_dict, f)

if __name__ == "__main__":
    test()