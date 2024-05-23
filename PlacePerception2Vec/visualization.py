import torch
import numpy as np
from SegmentationHead import SegWrapForViT
from lora import ViTWithLoRA
import timm


def semantic_inference(image, head_path, lora_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit = timm.create_model('vit_small_patch16_384.augreg_in21k_ft_in1k', pretrained=True)
    vit_with_lora = ViTWithLoRA(vit, 8, 1.0)
    vit_with_lora.load_lora_parameters(lora_path)
    seg_vit = SegWrapForViT(vit_with_lora, 384, 16, 384, 19).to(device)
    seg_vit.load_head_weight(head_path)
    seg_vit.eval()
    with torch.no_grad():
        seg_pred = seg_vit(image)
    seg_pred = seg_pred.squeeze()
    seg_pred = seg_pred.argmax(axis=0)
    seg_pred = seg_pred.numpy().astype(np.uint8)
    return seg_pred
