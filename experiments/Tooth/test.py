import argparse
import os.path as osp
import pdb
import time
import torch

import numpy as np
import pickle
from areconv.engine import SingleTester
from areconv.utils.torch import release_cuda
from areconv.utils.common import ensure_dir, get_log_string

from dataset import test_data_loader
from config import make_cfg
from model import create_model
from loss import Evaluator


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', default='Tooth', choices=['Tooth', 'val'], help='test benchmark')
    return parser


class Tester(SingleTester):
    def __init__(self, cfg):
        super().__init__(cfg, parser=make_parser())

        self.results = []
        # dataloader
        start_time = time.time()
        data_loader, neighbor_limits = test_data_loader(cfg, self.args.benchmark)
        loading_time = time.time() - start_time
        message = f'Data loader created: {loading_time:.3f}s collapsed.'
        self.logger.info(message)
        message = f'Calibrate neighbors: {neighbor_limits}.'
        self.logger.info(message)
        self.register_loader(data_loader)

        # model
        model = create_model(cfg).cuda()
        total = sum([param.nelement() for param in model.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))

        self.register_model(model)
        # evaluator
        self.evaluator = Evaluator(cfg).cuda()

        # preparation
        self.output_dir = osp.join(cfg.feature_dir, self.args.benchmark)
        ensure_dir(self.output_dir)

    def test_step(self, iteration, data_dict):
        output_dict = self.model(data_dict)
        return output_dict

    def eval_step(self, iteration, data_dict, output_dict):
        result_dict = self.evaluator(output_dict, data_dict)
        result_dict = {key: value.item() if isinstance(value, torch.Tensor) else value
                       for key, value in result_dict.items()}
        self.results.append(result_dict)  # 累积结果
        return result_dict

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        message = get_log_string(result_dict=result_dict)
        return message


def main():
    cfg = make_cfg()
    tester = Tester(cfg)
    tester.run()

    # 保存结果到 Excel
    output_path = "test_results.xlsx"
    tester.evaluator.save_results_to_excel(tester.results, file_path=output_path)

if __name__ == '__main__':
    main()
