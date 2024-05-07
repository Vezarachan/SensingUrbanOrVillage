import random
from PIL import ImageFilter, Image
from torch.utils.data import Dataset
import glob
import os


class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur:
    def __init__(self, sigma=(0.1, 2.)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class UnlabeledStreetViewImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.transform = transform
        self.file_list = glob.glob(os.path.join(image_dir, '*.jpg'))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img



