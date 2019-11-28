import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


__dict__ = {
    'common': CriterionDSN,
    'ohem': CriterionOhemDSN,
}

weight = torch.FloatTensor([
    0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969,
    0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843,
    1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_index=255, reduction='elementwise_mean', 
                 thresh=0.7, min_kept=100000, use_weight=False):
        super(OhemCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            self.criterion = CrossEntropyLoss(
                weight=weight, ignore_index=ignore_index, reduction=reduction)
        else:
            self.criterion = CrossEntropyLoss(
                ignore_index=ignore_index, reduction=reduction)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(1 - valid_mask, 1)
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(1 - valid_mask, self.ignore_index)
        target = target.view(b, h, w)

        return self.criterion(pred, target)


class CriterionDSN(CrossEntropyLoss):
    def __init__(self, ignore_index=255, reduction='elementwise_mean',
                 aux_weight=0.4, use_weight=False):
        if use_weight:
            super(CriterionDSN, self).__init__(weight=weight,
                                               ignore_index=ignore_index)
        else:
            super(CriterionDSN, self).__init__(ignore_index=ignore_index)

    def forward(self, preds, target):
        loss1 = self.criterion1(preds[0], target)
        loss = loss1
        losses = dict(loss=loss1.item())

        if self.aux_weight != 0. and preds[1] is not None:
            loss2 = self.criterion2(preds[1], target)
            loss += loss2 * self.aux_weight
            losses.update(dict(aux_loss=loss2.item())
        return loss, losses


class CriterionOhemDSN(nn.Module):
    '''
    DSN : We need to consider two supervision for the models.
    '''
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000,
                 reduction='elementwise_mean', aux_weight=0.4,
                 use_weight=False):
        super(CriterionOhemDSN, self).__init__()
        self.aux_weight = aux_weight
        self.criterion1 = OhemCrossEntropy(ignore_index, reduction, thresh, 
                                           min_kept, use_weight)
        self.criterion2 = CrossEntropyLoss(ignore_index=ignore_index, 
                                           reduction=reduction)

    def forward(self, preds, target):
        loss1 = self.criterion1(preds[0], target)
        loss = loss1
        losses = dict(loss=loss1.item())

        if self.aux_weight != 0. and preds[1] is not None:
            loss2 = self.criterion2(preds[1], target)
            loss += loss2 * self.aux_weight
            losses.update(dict(aux_loss=loss2.item())
        return loss, losses


def build_criterion(crit_config):
    __dict__ = {
        'common': CriterionDSN,
        'ohem': CriterionOhemDSN,
    }
    crit_cls = __dict__[crit_config['type']]
    crit = crit_cls(**crit_config.get('args', dict()))
    return crit 
