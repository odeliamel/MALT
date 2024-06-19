import torch
import numpy as np
from targeting import get_k_best_target_classes


class SmartTargetedAttack:
    def __init__(self, attack, k=3, norm=2, testk=None):
        self.attack = attack
        self.topk = k
        self.norm = norm
        self.testk = testk

    def attack_batch(self, model, X, y, batch_size=50):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        adv_total = X.detach().clone().cuda()
        ebt_targets = get_k_best_target_classes(model=model, data=X.detach(), target=y.detach(), ord=self.norm,
                                                topk=self.topk, testk=self.testk, batch_size=batch_size)
        failed_attacks_total = torch.ones(ebt_targets.shape[1]+1, X.shape[0], dtype=torch.bool).cuda()
        adv_total_targets = ebt_targets[:, 0].detach().clone()

        for i in range(ebt_targets.shape[1]):
            adv_target = ebt_targets[:, i]
            data, target = X.to(device), y.to(device)
            current_to_improve = failed_attacks_total[i].clone().type(torch.bool)
            X_adv = \
                self.attack(model, data[current_to_improve].detach(), adv_target[current_to_improve].detach()).detach()

            acc_each = (model(X_adv).detach().data.max(1)[1] == adv_target[current_to_improve].data).detach()
            current_improvement = torch.zeros_like(current_to_improve)
            current_improvement[current_to_improve] = acc_each

            adv_total[current_improvement] = X_adv[acc_each].detach()
            adv_total_targets[current_improvement] = adv_target[current_improvement].data.detach()

            failed_attacks_total[i+1] = failed_attacks_total[i] & ~current_improvement
            if failed_attacks_total[i+1].sum() == 0:
                break

            return adv_total, adv_total_targets, failed_attacks_total

    def test_w_data(self, model, x_orig, y_orig, num_restarts=1, batch_size=50):
        condition = lambda classification, t: classification == t
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        total_failed_attacks = torch.ones(x_orig.shape[0], device=device, dtype=torch.bool)

        inc_failed_attacks_total = torch.tensor([]).cuda()
        n_batches = int(np.ceil(x_orig.shape[0] / batch_size))

        for batch_idx in range(n_batches):
            bstart = batch_idx * batch_size
            bend = min((batch_idx + 1) * batch_size, x_orig.shape[0])
            data, target = x_orig[bstart:bend].clone().to(device), y_orig[bstart:bend].clone().to(device)
            for c in range(num_restarts):
                print("restart number ", c, "batch number", batch_idx, "(start {}, end {})".format(bstart, bend))

                current_to_improve = total_failed_attacks[bstart:bend].clone().type(torch.bool)

                X_adv, batch_targets, inc_failed_attacks = \
                    self.attack_batch(model, data[current_to_improve], target[current_to_improve],
                                                  batch_size=batch_size)

                inc_failed_attacks_total = torch.cat((inc_failed_attacks_total, inc_failed_attacks), dim=1)
                acc_each = condition(model(X_adv).data.max(1)[1], batch_targets).cuda()
                total_failed_attacks[bstart:bend][current_to_improve] = (
                            total_failed_attacks[bstart:bend][current_to_improve] * ~acc_each)

        asr = (~total_failed_attacks).sum().float() / total_failed_attacks.shape[0]
        print('asr_total: {:.2%}'.format(asr))
        model.zero_grad()
        inc_failed_sums = torch.tensor([]).cuda()
        for i in range(inc_failed_attacks_total.shape[0]):
            inc_failed_sums = torch.cat((inc_failed_sums, inc_failed_attacks_total[i].sum().unsqueeze(0)), dim=0)
        print("inc_failed_sums", inc_failed_sums)
        return asr