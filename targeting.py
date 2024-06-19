# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import torch.nn as nn

from utils import to_one_hot


def fast_find_matrix_around_image(images, targets, inxs, model, num_classes=1000, batch_size=50):
    print("find matrix")
    images_repeated = torch.repeat_interleave(images.detach().clone().cpu(), inxs.shape[1], dim=0).detach()

    eye = to_one_hot(inxs, num_classes=num_classes)
    if targets is not None:
        targets_repeated = torch.repeat_interleave(targets, inxs.shape[1], dim=0).detach()
        eye = eye.scatter(dim=1, index=targets_repeated.unsqueeze(1), value=-1)
    eye = eye.type(torch.FloatTensor)

    resmatrices = batch_gradient_w_loss(model, images_repeated, eye, batch_size=batch_size)
    resmatrices = resmatrices.reshape((images.shape[0], inxs.shape[-1],
                                        images.shape[-3] * images.shape[-2] * images.shape[-1]))

    return resmatrices


def batch_gradient_w_loss(model, input, backward_var, batch_size=50):
    print("batch_gradient_w_loss")
    grads = torch.tensor([])
    start = 0
    while start < input.shape[0]:
        end = min(start + batch_size, input.shape[0])
        batch_inputs = input[start:end].cuda()
        batch_backward_var = backward_var[start:end].cuda()
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            batch_inputs = nn.Parameter(batch_inputs)
            batch_inputs.requires_grad_(True)
            output = model(batch_inputs)
            output.backward(batch_backward_var)
            g = batch_inputs.grad.data.detach().clone().cpu()
            grads = torch.cat((grads, g), dim=0)
            batch_inputs.grad.zero_()
            batch_inputs.detach()

        start = end

    return grads


def get_class_rank(x, next_topk_inxs, ord, linmatrix, srcoutput, srcclass, type=torch.float):
    print("get_class_rank")
    epsilon_matrix = torch.zeros((x.shape[0], next_topk_inxs.shape[-1]), dtype=type)#.cuda()

    src_classes_scores = srcoutput[range(srcoutput.shape[0]), srcclass.view(-1)].unsqueeze(1)
    score_diffs = torch.repeat_interleave(src_classes_scores, next_topk_inxs.shape[-1], dim=1) - \
                  srcoutput[torch.arange(srcoutput.shape[0]).unsqueeze(-1), next_topk_inxs]
    matrix_raw_diffs = linmatrix#.cuda()

    if ord == 2:
        matrix_raw_diff_norms = torch.norm(matrix_raw_diffs, p=2, dim=2)#.unsqueeze(0)

    if ord == np.inf:
        matrix_raw_diff_norms = torch.norm(matrix_raw_diffs, p=1, dim=2)#.unsqueeze(0)

    non_zero_row_mask = (matrix_raw_diff_norms != 0)

    if non_zero_row_mask.sum() != 0:
        epsilon_matrix[non_zero_row_mask] = \
            score_diffs[non_zero_row_mask].cpu() / ((matrix_raw_diff_norms[non_zero_row_mask]).cpu())

    all_classes_matrix = np.inf * torch.ones(x.shape[0], srcoutput.shape[-1])#.cuda()
    all_classes_matrix[torch.arange(x.shape[0]).unsqueeze(-1), next_topk_inxs] = epsilon_matrix
    return all_classes_matrix


def get_k_best_target_classes(model, data, target, ord, topk, testk=None, batch_size=50):
    with torch.no_grad():
        srcoutput = model(data)
    _, srcclass = torch.max(srcoutput, dim=1)
    n_srcoutput = srcoutput.scatter(dim=1, index=target.unsqueeze(1), value=-float("inf"))
    if testk == None or testk > srcoutput.shape[-1]-1:
        testk = srcoutput.shape[-1]-1

    _, next_topk_inxs = torch.topk(n_srcoutput, k=testk, dim=1)
    if topk > srcoutput.shape[-1]-1:
        topk = srcoutput.shape[-1]-1

    linmatrix = fast_find_matrix_around_image(data, target, next_topk_inxs, model, num_classes=srcoutput.shape[-1], batch_size=batch_size)
    epsilon_matrix = get_class_rank(data, next_topk_inxs, ord, linmatrix, srcoutput, srcclass, type=torch.float)
    return epsilon_matrix.topk(dim=1, largest=False, k=topk)[1].cuda()


