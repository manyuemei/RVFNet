from __future__ import absolute_import

import torch
from torch import nn
import os
import csv


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Args:
        margin (float): Margin for triplet.
        epoch (int): The number of epochs. Default is None.
        path (str): The path where fingerprints and Euclidean distances between fingerprints are stored. Default is None.
        val (bool): Determines whether in the validation stage. Default is False.
    """
    def __init__(self, margin, epoch=None, path=None, val=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.val = val
        self.path = path
        self.epoch = epoch

    def forward(self, inputs, targets):
        """
        Args:
            inputs: visualization_feature_map matrix with shape (batch_size, fingerprint_length)#69x64
            targets: labels of videos with shape (batch_size, 1)#69x1
        """
        n = inputs.size(0)
        hash_p, hash_n = [], []
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)

        dist = dist + dist.t()
        dist.addmm_(beta=1, mat1=inputs.to(torch.float32), mat2=inputs.t().to(torch.float32), alpha=-2)

        if self.val:
            dist1 = dist.sqrt()
            for k in range(n):
                for z in range(k + 1, n):
                    if mask[k][z]:
                        hash_p.append(dist1[k][z])
                    else:
                        hash_n.append(dist1[k][z])

            hash_p.append('*')
            hash_p.extend(hash_n)

            with open(os.path.join(self.path, "epoch_" + str(self.epoch) + "distance.csv"), 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                row_to_write = ['{:.4f}'.format(single_code) if torch.is_tensor(single_code) else '*' for single_code in
                                hash_p]
                writer.writerow(row_to_write)

            with open(os.path.join(self.path, "epoch_" + str(self.epoch) + "fingerprint.csv"), 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for i in range(n):
                    row_to_write = ['{:.4f}'.format(single_code) for single_code in inputs[i]]
                    writer.writerow(row_to_write)
            del hash_n,hash_p,dist1

        dist1 = dist.clamp(min=1e-6).sqrt()

        for j in range(n):
            dist_ap.append(dist1[j][mask[j]].max().unsqueeze(0))
            dist_an.append(dist1[j][mask[j] == 0].min().unsqueeze(0))

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        y = torch.ones_like(dist_an)

        loss = self.ranking_loss(dist_an, dist_ap, y)

        return loss

if __name__ == '__main__':
    # test
    use_gpu = False
    model = TripletLoss(1)
    features = torch.Tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    label = torch.Tensor([0, 1, 1]).long()
    loss = model(features, label)
