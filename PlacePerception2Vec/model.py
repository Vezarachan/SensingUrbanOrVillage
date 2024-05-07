import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplePatchifier(nn.Module):
    def __init__(self, patch_size=16):
        super(SimplePatchifier, self).__init__()
        self.patch_size = patch_size

    def forward(self, x):
        B, C, H, W = x.shape
        x = ((x.permute(0, 2, 3, 1)
             .unfold(1, self.patch_size, self.patch_size)
             .unfold(2, self.patch_size, self.patch_size))
             .contiguous()
             .view(B, -1, C, self.patch_size, self.patch_size))
        return x


class TwoLayerNN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super(TwoLayerNN, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.layer = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, out_features)
        )

    def forward(self, x):
        return self.layer(x) + x


class ViGBlock(nn.Module):
    def __init__(self, in_features, num_edges=9, head_num=1):
        super(ViGBlock, self).__init__()
        self.k = num_edges
        self.num_edges = num_edges
        self.in_layer1 = TwoLayerNN(in_features)
        self.out_layer1 = TwoLayerNN(in_features)
        self.droppath1 = nn.Identity()
        self.in_layer2 = TwoLayerNN(in_features, in_features * 4)
        self.out_layer2 = TwoLayerNN(in_features, in_features * 4)
        self.droppath2 = nn.Identity()
        self.multi_head_fc = nn.Conv1d(in_features * 2, in_features, 1, 1, groups=head_num)

    def forward(self, x):
        B, N, C = x.shape
        sim = x @ x.transpose(-1, -2)
        graph = sim.topk(self.k, dim=-1).indices

        shortcut = x
        x = self.in_layer1(x.view(B * N, -1)).view(B, N, -1)

        # aggregation
        neighbor_features = x[torch.arange(B).unsqueeze(-1).expand(-1, N).unsqueeze(-1), graph]
        x = torch.stack([x, (neighbor_features - x.unsqueeze(-2)).amax(dim=-2)], dim=-1)

        # update
        # multi head
        x = self.multi_head_fc(x.view(B * N, -1, 1)).view(B, N, -1)
        x = self.droppath1(self.out_layer1(F.gelu(x).view(B * N, -1)).view(B, N, -1))
        x = x + shortcut

        x = self.droppath2(self.out_layer2(F.gelu(x).view(B * N, -1)).view(B, N, -1))
        return x


class VisionGNN(nn.Module):
    def __init__(self, in_features=3 * 16 * 16,
                 out_features=320,
                 num_patches=196,
                 num_vig_blocks=16,
                 num_edges=9,
                 head_num=1,
                 num_classes=1024):
        super(VisionGNN, self).__init__()
        self.patchifier = SimplePatchifier(patch_size=16)
        self.patch_embedding = nn.Sequential(
            nn.Linear(in_features, out_features//2),
            nn.BatchNorm1d(out_features//2),
            nn.GELU(),
            nn.Linear(out_features//2, out_features//4),
            nn.BatchNorm1d(out_features//4),
            nn.GELU(),
            nn.Linear(out_features//4, out_features//8),
            nn.BatchNorm1d(out_features//8),
            nn.GELU(),
            nn.Linear(out_features//8, out_features//4),
            nn.BatchNorm1d(out_features//4),
            nn.GELU(),
            nn.Linear(out_features//4, out_features//2),
            nn.BatchNorm1d(out_features//2),
            nn.GELU(),
            nn.Linear(out_features//2, out_features),
            nn.BatchNorm1d(out_features),
        )
        self.pose_embedding = nn.Parameter(torch.rand(num_patches, out_features))
        self.blocks = nn.Sequential(
            *[ViGBlock(out_features, num_edges, head_num) for _ in range(num_vig_blocks)]
        )
        self.fc = nn.Linear(out_features * num_patches, num_classes)

    def forward(self, x):
        x = self.patchifier(x)
        B, N, C, H, W = x.shape
        x = self.patch_embedding(x.view(B * N, -1)).view(B, N, -1)
        x = x + self.pose_embedding
        x = self.blocks(x)
        x = self.fc(x.view(B, -1))
        return x


class MoCo(nn.Module):
    def __init__(self, base_encoder, dim=128, queue_size=65536, t=0.07, m=0.999):
        super(MoCo, self).__init__()
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)
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
        # f_mem = F.normalize(f_mem, dim=1)

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
