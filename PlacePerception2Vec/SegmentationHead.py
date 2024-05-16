from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from lora import ViTWithLoRA
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision.transforms as T


def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)


class SegWrapForViT(nn.Module):
    def __init__(self, vit_model: ViTWithLoRA, image_size: int, patches: int, dim: int, num_classes: int):
        super(SegWrapForViT, self).__init__()
        self.vit = vit_model
        self.deeplab_head = DeepLabHead(dim, num_classes)
        h, w = as_tuple(image_size)
        fh, fw = as_tuple(patches)
        self.gh, self.gw = h // fh, w // fw
        self.h, self.w = h, w
        self.resize = T.Resize((h, w))

    def forward(self, x):
        _, _, img_h, img_w = x.shape
        x = self.resize(x)
        x = self.vit(x)
        b, gh_gw, d = x.shape
        x = x[:, :-1, :]
        x = x.transpose(1, 2)
        x = x.reshape(b, d, self.gh, self.gw)
        x = self.deeplab_head(x)
        x = F.interpolate(x, size=(img_h, img_w), mode='bilinear', align_corners=False)
        return x

    def save_head_weight(self, filename):
        torch.save(self.deeplab_head.state_dict(), filename)

    def load_head_weight(self, filename):
        self.deeplab_head.load_state_dict(torch.load(filename))


if __name__ == '__main__':
    img = torch.randn(3, 3, 1024, 2048)
    model = timm.create_model('vit_small_patch16_384.augreg_in21k_ft_in1k', pretrained=True)
    vit_with_lora = ViTWithLoRA(model, 8, 1.0)
    seg_vit = SegWrapForViT(vit_model=vit_with_lora, image_size=384,
                            patches=16, dim=384, num_classes=19)
    mask = seg_vit(img)
    num_params = sum(p.numel() for p in seg_vit.parameters() if p.requires_grad)
    print(mask.shape)
    print(num_params)



