import torch
import torch.nn as nn
import torch.nn.functional as F
from util.tensor_util import compute_tensor_iu

from collections import defaultdict
import numpy as np


def get_iou_hook(values):
    return 'iou/iou', (values['hide_iou/i']+1)/(values['hide_iou/u']+1)

def get_sec_iou_hook(values):
    return 'iou/sec_iou', (values['hide_iou/sec_i']+1)/(values['hide_iou/sec_u']+1)

iou_hooks_so = [
    get_iou_hook,
]

iou_hooks_mo = [
    get_iou_hook,
    get_sec_iou_hook,
]


class BootstrappedCE(nn.Module):
    def __init__(self, start_warm=20000, end_warm=70000, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p

    def forward(self, input, target, it):
        if it < self.start_warm:
            return F.cross_entropy(input, target), 1.0

        raw_loss = F.cross_entropy(input, target, reduction='none').view(-1)
        num_pixels = raw_loss.numel()

        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1-self.top_p)*((self.end_warm-it)/(self.end_warm-self.start_warm))
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        return loss.mean(), this_p

def dice_loss_func(input, target):
    smooth = 1.
    n = input.size(0)
    iflat = input.view(n, -1)
    tflat = target.view(n, -1)
    intersection = (iflat * tflat).sum(1)
    loss = 1 - ((2. * intersection + smooth) /
                (iflat.sum(1) + tflat.sum(1) + smooth))
    return loss.mean()



class LossComputer:
    def __init__(self, para):
        super().__init__()
        self.para = para
        self.bce = BootstrappedCE(start_warm=self.para['start_warm'], end_warm=self.para['end_warm'])


    def compute(self, data, it):
        losses = defaultdict(int)

        b, s, _, _, _ = data['gt'].shape
        selector = data.get('selector', None)


        for i in range(1, s):
            # Have to do it in a for-loop like this since not every entry has the second object
            # Well it's not a lot of iterations anyway
            for j in range(b):
                if selector is not None and selector[j][1] > 0.5:
                    loss1, p = self.bce(data['logits_%d'%i][j:j+1], data['cls_gt'][j:j+1,i], it)

                    # Boundary Loss
                    boundary_logits1 = data['boundary_%d' % i][j:j + 1, 0:1]
                    boundary_logits2 = data['boundary_%d' % i][j:j + 1, 1:2]
                    laplacian_kernel = torch.tensor(
                        [-1, -1, -1, -1, 8, -1, -1, -1, -1],
                        dtype=torch.float32, device=boundary_logits1.device).reshape(1, 1, 3, 3).requires_grad_(False)
                    boundary_targets1 = F.conv2d(data['gt'][j:j+1, i].float(), laplacian_kernel, padding=1)
                    boundary_targets2 = F.conv2d(data['sec_gt'][j:j + 1, i].float(), laplacian_kernel, padding=1)
                    boundary_targets1 = boundary_targets1.clamp(min=0)
                    boundary_targets2 = boundary_targets2.clamp(min=0)
                    boundary_targets1[boundary_targets1 > 0.1] = 1
                    boundary_targets1[boundary_targets1 <= 0.1] = 0
                    boundary_targets2[boundary_targets2 > 0.1] = 1
                    boundary_targets2[boundary_targets2 <= 0.1] = 0
                    if boundary_logits1.shape[-1] != boundary_targets1.shape[-1]:
                        boundary_targets1 = F.interpolate(
                            boundary_targets1, boundary_logits1.shape[2:], mode='bilinear')
                    if boundary_logits2.shape[-1] != boundary_targets2.shape[-1]:
                        boundary_targets2 = F.interpolate(
                            boundary_targets2, boundary_logits2.shape[2:], mode='bilinear')

                    bce_loss1 = F.binary_cross_entropy_with_logits(boundary_logits1, boundary_targets1)
                    dice_loss1 = dice_loss_func(torch.sigmoid(boundary_logits1), boundary_targets1)
                    bce_loss2 = F.binary_cross_entropy_with_logits(boundary_logits2, boundary_targets2)
                    dice_loss2 = dice_loss_func(torch.sigmoid(boundary_logits2), boundary_targets2)
                    loss2 = 0.5 * (bce_loss1 + dice_loss1 + bce_loss2 + dice_loss2)

                else:
                    loss1, p = self.bce(data['logits_%d'%i][j:j+1,:2], data['cls_gt'][j:j+1,i], it)


                    # Boundary Loss
                    boundary_logits = data['boundary_%d' % i][j:j + 1]
                    laplacian_kernel = torch.tensor(
                        [-1, -1, -1, -1, 8, -1, -1, -1, -1],
                        dtype=torch.float32, device=boundary_logits.device).reshape(1, 1, 3, 3).requires_grad_(False)
                    boundary_targets = F.conv2d(data['gt'][j:j+1, i].float(), laplacian_kernel, padding=1)
                    boundary_targets = boundary_targets.clamp(min=0)
                    boundary_targets[boundary_targets > 0.1] = 1
                    boundary_targets[boundary_targets <= 0.1] = 0
                    if boundary_logits.shape[-1] != boundary_targets.shape[-1]:
                        boundary_targets = F.interpolate(
                            boundary_targets, boundary_logits.shape[2:], mode='bilinear')

                    bce_loss = F.binary_cross_entropy_with_logits(boundary_logits[:, 0:1], boundary_targets)
                    dice_loss = dice_loss_func(torch.sigmoid(boundary_logits[:, 0:1]), boundary_targets)
                    loss2 = bce_loss + dice_loss




                losses['loss_%d'%i] = losses['loss_%d'%i] + loss1 / b + 0.25 * loss2 / b
                losses['p'] += p / b / (s-1)



            losses['total_loss'] += losses['loss_%d'%i]


            new_total_i, new_total_u = compute_tensor_iu(data['mask_%d'%i]>0.5, data['gt'][:,i]>0.5)
            losses['hide_iou/i'] += new_total_i
            losses['hide_iou/u'] += new_total_u

            if selector is not None:
                new_total_i, new_total_u = compute_tensor_iu(data['sec_mask_%d'%i]>0.5, data['sec_gt'][:,i]>0.5)
                losses['hide_iou/sec_i'] += new_total_i
                losses['hide_iou/sec_u'] += new_total_u

        return losses
