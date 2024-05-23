from functools import partial
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer as ViT
from torch.nn.parameter import Parameter
from safetensors import safe_open
from safetensors.torch import save_file
import timm


class LoRALayer(nn.Module):
    def __init__(self, in_channels, out_channels, rank, alpha):
        super(LoRALayer, self).__init__()
        self.std = torch.sqrt(torch.tensor(rank).float())
        self.W_a = nn.Linear(in_channels, rank)
        self.W_b = nn.Linear(rank, out_channels)
        self.alpha = alpha
        self.rank = rank
        self.reset_parameters()

    def forward(self, x):
        x = self.alpha / self.std * self.W_b(self.W_a(x))
        return x

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.W_b.weight)


class QkvWithLoRA(nn.Module):
    def __init__(self, qkv: nn.Module, rank: int, alpha: int):
        super(QkvWithLoRA, self).__init__()
        self.dim = qkv.in_features
        self.qkv = qkv
        self.lora_q = LoRALayer(self.dim, self.dim, rank, alpha)
        self.lora_k = LoRALayer(self.dim, self.dim, rank, alpha)

    def forward(self, x):
        qkv = self.qkv(x)
        qkv[:, :, :self.dim] += self.lora_q(x)
        qkv[:, :, -self.dim:] += self.lora_k(x)
        return qkv


class ViTWithLoRA(nn.Module):
    def __init__(self, vit: ViT, rank: int, alpha: int):
        super(ViTWithLoRA, self).__init__()
        self.vit = vit
        assert rank > 0
        assert alpha > 0
        assign_lora = partial(QkvWithLoRA, rank=rank, alpha=alpha)
        for i, block in enumerate(vit.blocks):
            block.attn.qkv = assign_lora(block.attn.qkv)

        for param in vit.parameters():
            param.requires_grad = False

        for block in vit.blocks:
            for param in block.attn.qkv.lora_q.parameters():
                param.requires_grad = True
            for param in block.attn.qkv.lora_k.parameters():
                param.requires_grad = True

        for param in vit.head.parameters():
            param.requires_grad = True

    def save_lora_parameters(self, filename):
        assert filename.endswith(".safetensors")
        q_a_tensors = {f"w_q_a_{i:03d}": block.attn.qkv.lora_q.W_a.weight for i, block in enumerate(self.vit.blocks)}
        k_a_tensors = {f"w_k_a_{i:03d}": block.attn.qkv.lora_k.W_a.weight for i, block in enumerate(self.vit.blocks)}
        q_b_tensors = {f"w_q_b_{i:03d}": block.attn.qkv.lora_q.W_b.weight for i, block in enumerate(self.vit.blocks)}
        k_b_tensors = {f"w_k_b_{i:03d}": block.attn.qkv.lora_k.W_b.weight for i, block in enumerate(self.vit.blocks)}
        # _in = self.vit.head.in_features
        # _out = self.vit.head.out_features
        # fc_tensors = {f"fc_{_in}in_{_out}out": self.vit.head.weight}
        merge_dict = {**q_a_tensors, **k_a_tensors, **q_b_tensors, **k_b_tensors}
        save_file(merge_dict, filename)

    def load_lora_parameters(self, filename):
        assert filename.endswith(".safetensors")
        with safe_open(filename, framework="pt") as f:
            for i, block in enumerate(self.vit.blocks):
                saved_q_a_key = f"w_q_a_{i:03d}"
                saved_q_a_tensor = f.get_tensor(saved_q_a_key)
                block.attn.qkv.lora_q.W_a.weight = Parameter(saved_q_a_tensor)
                saved_k_a_key = f"w_k_a_{i:03d}"
                saved_k_a_tensor = f.get_tensor(saved_k_a_key)
                block.attn.qkv.lora_k.W_a.weight = Parameter(saved_k_a_tensor)
                saved_q_b_key = f"w_q_b_{i:03d}"
                saved_q_b_tensor = f.get_tensor(saved_q_b_key)
                block.attn.qkv.lora_q.W_b.weight = Parameter(saved_q_b_tensor)
                saved_k_b_key = f"w_k_b_{i:03d}"
                saved_k_b_tensor = f.get_tensor(saved_k_b_key)
                block.attn.qkv.lora_k.W_b.weight = Parameter(saved_k_b_tensor)
            # _in = self.vit.head.in_features
            # _out = self.vit.head.out_features
            # saved_key = f"fc_{_in}in_{_out}out"
            # try:
            #     saved_tensor = f.get_tensor(saved_key)
            #     self.vit.head.weight = Parameter(saved_tensor)
            # except ValueError:
            #     print("this fc weight is not for this model")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit.forward_features(x)


if __name__ == '__main__':
    model = timm.create_model('vit_small_patch16_384.augreg_in21k_ft_in1k', pretrained=True)
    vit_with_lora = ViTWithLoRA(model, 8, 1.0)
    vit_with_lora.save_lora_parameters("./lora_parameters.safetensors")
    vit_with_lora.load_lora_parameters('./lora_parameters.safetensors')
