import argparse
import logging
import os

# from attacks.auto_pgd import APGD
from adv_lib.attacks.auto_pgd import apgd

from utils import log_success_indices

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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
parser.add_argument('--batch_size', type=int, default="10")

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
# model = nn.DataParallel(model)


# Since the original targeted APGD perturb function is not a targeted attack - it chooses top 9 targets.
# we use jeromerony's implementation to targeted APGD: adversarial-library
# (from https://github.com/jeromerony/adversarial-library.git)
# This attack function implements a simple targeted attack towards target y.


base_attack = lambda m, x_orig, y: apgd(model=m, inputs=x_orig, labels=y,
                                        eps=args.eps, norm=np.inf, targeted=True)

malt_attack = MALTAttack(attack_func=base_attack, k=9, norm=np.inf, testk=100)

x_adv, _ = malt_attack.test_w_data(model, x_orig=clean_x_test, y_orig=clean_y_test,
                                   batch_size=args.batch_size, logger=logger)

log_success_indices(clean_x_test=clean_x_test, clean_y_test=clean_y_test,
                    device=args.device, logger=logger, model=model, x_adv=x_adv)
