import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import shutil
from torchvision.transforms import transforms as T
import timm
from tqdm import tqdm
import numpy as np
from metrics import Evaluator
from SegmentationHead import SegWrapForViT
from lora import ViTWithLoRA
from torch.utils.data import DataLoader
from cityscapes import Cityscapes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_miou = 0.0


def train_step(model, epoch, criterion, optimizer, data_loader, args):
    running_loss = []
    model.train()
    train_bar = tqdm(data_loader)
    for i, sample in enumerate(train_bar):
        image, label = sample[0].to(device), sample[1].long().to(device)
        optimizer.zero_grad()
        pred = model(image)
        label = torch.squeeze(label, 1)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        running_loss.append(loss.item())
        train_bar.set_description(
            f'Epoch: [{epoch}] Loss: {round(sum(running_loss) / len(running_loss), 6)}')

    model.save_head_weight(f'{args.weight_path}/head_epoch_{epoch}.pth')
    model.vit.save_lora_parameters(f'{args.weight_path}/lora_epoch_{epoch}.safetensors')


def validation_step(epoch, model, criterion, optimizer, data_loader, evaluator, args):
    correct, total = 0, 0
    model.eval()

    validation_bar = tqdm(data_loader)
    evaluator.reset()
    test_loss = 0.0
    for i, sample in enumerate(validation_bar):
        image, label = sample[0].to(device), sample[1].long().to(device)
        with torch.no_grad():
            output = model(image)
        loss = criterion(output, label)
        test_loss += loss.item()
        validation_bar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
        pred = output.data.cpu().numpy()
        label = torch.squeeze(label, 1)
        label = label.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        evaluator.add_batch(label, pred)

        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        print(f'Acc:{Acc}, Acc_class:{Acc_class}, mIoU:{mIoU}')
        print(f'Loss: {test_loss}')

        print(f'best miou: {best_miou}')

        if mIoU > best_miou:
            is_best = True
            best_miou = mIoU
            model.save_head_weight(f'{args.weight_path}/head_best.pth')
            model.vit.save_lora_parameters(f'{args.weight_path}/lora_best.safetensors')


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


def main():
    parser = argparse.ArgumentParser('LoRA ViT for Semantic Segmentation')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                    training (default: auto)')
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--data', type=str, default=None, help='path to dataset')
    parser.add_argument('--weight-path', type=str, default=None, help='path to saved weights')

    args = parser.parse_args()

    vit = timm.create_model('vit_small_patch16_384.augreg_in21k_ft_in1k', pretrained=True)
    lora_model = ViTWithLoRA(vit, 8, 1.0)
    model = SegWrapForViT(lora_model, 384, 16, 384, 19).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=0, last_epoch=-1, verbose=False)
    train_transforms = T.Compose([
        T.ToTensor(),
        T.RandomHorizontalFlip()
    ])

    val_transforms = T.Compose([
        T.ToTensor()
    ])
    cityscapes_train = Cityscapes(args.data, split='train', transform=train_transforms)
    cityscapes_val = Cityscapes(args.data, split='val', transform=val_transforms)
    train_loader = DataLoader(cityscapes_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(cityscapes_val, batch_size=args.batch_size, shuffle=False, num_workers=2)
    evaluator = Evaluator(num_class=19)

    for epoch in range(args.epochs):
        train_step(model, epoch, criterion, optimizer, train_loader, args)
        if epoch % args.eval_interval == (args.eval_interval - 1):
            validation_step(epoch, model, criterion, optimizer, val_loader, evaluator, args)
        scheduler.step()


if __name__ == '__main__':
    # dataset = CityscapesDataset('D:\\Research\\datasets\\cityscapes', split='val')
    # print(dataset[0]['label'].unique())
    main()
