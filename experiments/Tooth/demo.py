import argparse
import pickle

import torch
import numpy as np

from areconv.utils.data import registration_collate_fn_stack_mode, precompute_neibors
from areconv.utils.torch import to_cuda, release_cuda
from areconv.utils.open3d import make_open3d_point_cloud, get_color, draw_geometries
from areconv.utils.registration import compute_registration_error

from config import make_cfg
from model import create_model


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_file", default='../../data/demo/demo.pkl', help="path to pkl file containing src_pcd, tgt_pcd, gt_pose")
    parser.add_argument("--weights", default='../../pretrain/tooth.pth.tar', help="model weights file")

    return parser


def load_data(args):

    with open(args.pkl_file, 'rb') as f:
        demodata = pickle.load(f)

    src_points = demodata['src_pcd']   # [N, 3]
    ref_points = demodata['tgt_pcd']  # [M, 3]
    transform = demodata['gt_pose']   # [4, 4]

    src_feats = np.ones_like(src_points[:, :1])
    ref_feats = np.ones_like(ref_points[:, :1])

    data_dict = {
        "ref_points": ref_points.astype(np.float32),
        "src_points": src_points.astype(np.float32),
        "ref_feats": ref_feats.astype(np.float32),
        "src_feats": src_feats.astype(np.float32),
        "transform": transform.astype(np.float32),
    }

    return data_dict


def main():
    parser = make_parser()
    args = parser.parse_args()

    cfg = make_cfg()

    # prepare data
    data_dict = load_data(args)
    data_dict = registration_collate_fn_stack_mode(
        [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.num_neighbors, cfg.backbone.subsample_ratio
    )

    # prepare model
    model = create_model(cfg).cuda()
    state_dict = torch.load(args.weights)
    model.load_state_dict(state_dict["model"])

    # prediction
    data_dict = to_cuda(data_dict)
    data = precompute_neibors(data_dict['points'], data_dict['lengths'],
                              cfg.backbone.num_stages,
                              cfg.backbone.num_neighbors,
                              )
    data_dict.update(data)
    output_dict = model(data_dict)
    data_dict = release_cuda(data_dict)
    output_dict = release_cuda(output_dict)

    # get results
    ref_points = output_dict["ref_points"]
    src_points = output_dict["src_points"]
    estimated_transform = output_dict["estimated_transform"]


    # visualization
    ref_pcd = make_open3d_point_cloud(ref_points)
    ref_pcd.estimate_normals()
    ref_pcd.paint_uniform_color(get_color("custom_yellow"))
    src_pcd = make_open3d_point_cloud(src_points)
    src_pcd.estimate_normals()
    src_pcd.paint_uniform_color(get_color("custom_blue"))
    draw_geometries(ref_pcd, src_pcd)
    src_pcd = src_pcd.transform(estimated_transform)
    draw_geometries(ref_pcd, src_pcd)

    # compute error
    transform = data_dict["transform"]
    rre, rte = compute_registration_error(transform, estimated_transform)
    print(f"RRE(deg): {rre:.3f}, RTE(m): {rte:.3f}")


if __name__ == "__main__":
    main()
