import torch
import torch.nn as nn
import torch.nn.functional as F


class PlacePerception2Vec(nn.Module):
    def __init__(self):
        super(PlacePerception2Vec, self).__init__()


class MoCo(nn.Module):
    def __init__(self, encoder_q, encoder_k, dim=128, queue_size=65536, t=0.07, m=0.999):
        super(MoCo, self).__init__()
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k
        self.queue_size = queue_size
        self.t = t
        self.m = m
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        self.register_buffer('queue', torch.randn(dim, self.queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _update_queue(self, keys):
        batch_size = keys.size(0)
        ptr = int(self.queue_ptr)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _shuffled_idx(self, batch_size):
        shuffled_idxs = torch.randperm(batch_size).long().cuda()
        reverse_idxs = torch.zeros(batch_size).long().cuda()
        value = torch.arange(batch_size).long().cuda()
        reverse_idxs.index_copy_(0, shuffled_idxs, value)
        return shuffled_idxs, reverse_idxs

    def InfoNCE_logits(self, f_q, f_k):
        f_k = f_k.detach()
        f_mem = self.queue.clone().detach()

        f_q = F.normalize(f_q, dim=1)
        f_k = F.normalize(f_k, dim=1)
        f_mem = F.normalize(f_mem, dim=1)

        l_pos = torch.einsum('nc,nc->n', [f_q, f_k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [f_q, f_mem]).unsqueeze(-1)

        logits = torch.cat((l_pos, l_neg), dim=1)
        logits /= self.t

        N = logits.shape[0]
        labels = torch.zeros(N, dtype=torch.long).cuda()
        return logits, labels

    def forward(self, q, k):
        batch_size = q.size(0)
        f_q = self.encoder_q(q)
        suffled_idxs, reverse_idxs = self._shuffled_idx(batch_size)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = k[suffled_idxs]
            f_k = self.encoder_k(k)
            f_k = f_k[reverse_idxs]

        logits, labels = self.InfoNCE_logits(f_q, f_k)

        self._update_queue()

        return logits, labels
