import argparse
import logging


import numpy as np
import torch
import torch.nn as nn


from robustbench.data import get_preprocessing, load_clean_dataset
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel

from MALT_attack import MALTAttack
from autoattack.autopgd_base import APGDAttack_targeted
from robustbench import load_model


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--n_examples', type=int, default="5000")
parser.add_argument('--dataset', default="imagenet")
parser.add_argument('--threat_model', default="Linf")
parser.add_argument('--model_name', default="Liu2023Comprehensive_Swin-L")
parser.add_argument('--data_dir', default="/net/mraid11/export/vision/datasets/ImageNet")
parser.add_argument('--batch_size', type=int, default="20")

parser.add_argument('--model_dir', default="./models")
parser.add_argument('--eps', type=float, default="0.01568627450980392")
parser.add_argument('--seed', type=int, default="2024")
parser.add_argument('--device', default="cuda")

args = parser.parse_args()

dataset_: BenchmarkDataset = BenchmarkDataset(args.dataset)
threat_model_: ThreatModel = ThreatModel(args.threat_model)


prepr = get_preprocessing(dataset_, threat_model_, args.model_name, None)

clean_x_test, clean_y_test = load_clean_dataset(dataset_, args.n_examples,
                                                args.data_dir, prepr)

model = load_model(model_name=args.model_name,
                   model_dir=args.model_dir,
                   dataset=args.dataset,
                   threat_model=args.threat_model)


model = model.to(args.device)


class APGDTargeted:
    def __init__(self):
        self.targeted = True
        self.base_attack = APGDAttack_targeted(model, n_restarts=1, n_iter=100, verbose=False,
                                          eps=args.eps, norm='Linf', eot_iter=1, rho=.75, seed=args.seed,
                                          device=args.device)

    def attack(self, model, x_orig, y):
        self.base_attack.init_hyperparam(x_orig)
        with torch.no_grad():
            output = model(x_orig)
        model.zero_grad()
        self.base_attack.y_target = y.detach().clone()
        y_new = output.max(1)[1].detach().clone().long().to(args.device)
        best_curr, acc_curr, loss_curr, adv_curr = self.base_attack.attack_single_run(x_orig, y_new)
        return adv_curr


malt_attack = MALTAttack(attack_func=APGDTargeted().attack, k=9, norm=np.inf, testk=100)

malt_attack.test_w_data(model, x_orig=clean_x_test, y_orig=clean_y_test, batch_size=args.batch_size, logger=logger)
