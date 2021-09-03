import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import glob
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import argparse
from utils.utils import AverageMeter
from utils.ca_warnup import *
from timeit import default_timer as timer
from datetime import timedelta
from torch_poly_lr_decay import PolynomialLRDecay
import torch.cuda.amp as amp

random_seed = 553459345
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(random_seed)

patch_size = 768


class ImageDataset(Dataset):
    def __init__(self, dataset_path, is_train=True):
        self.is_train = is_train
        if self.is_train:
            self.input_paths = sorted(glob.glob(os.path.join(dataset_path, "train_input_image_3200_only", "*")))+ sorted(glob.glob(os.path.join(dataset_path, "valid_input_image_3200_ol2", "*")))
            self.label_paths = sorted(glob.glob(os.path.join(dataset_path, "train_label_image_3200_only", "*")))+ sorted(glob.glob(os.path.join(dataset_path, "valid_label_image_3200_ol2", "*")))
        else:
            self.input_paths  = sorted(glob.glob(os.path.join(dataset_path, "valid_input_image_3200_only", "*")))
            self.label_paths = sorted(glob.glob(os.path.join(dataset_path, "valid_label_image_3200_only", "*")))

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, index):
        img1 = np.array(Image.open(self.input_paths[index]))
        img2 = np.array(Image.open(self.label_paths[index]))

        if np.random.rand() < 0.5 and self.is_train:
            img1 = img1[:, ::-1, :]
            img2 = img2[:, ::-1, :]

        image_name = "_".join(self.input_paths[index].split('/')[-1].split(".")[0].split("_")[:-2])
        y = int(self.input_paths[index].split('/')[-1].split(".")[0].split("_")[-1])
        x = int(self.input_paths[index].split('/')[-1].split(".")[0].split("_")[-2])
        return transforms.ToTensor()(img1.copy()), transforms.ToTensor()(img2.copy()), y, x, image_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='timm-efficientnet-b3')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--data_path', type=str, default='/home/leehanbeen/home/leehanbeen/PycharmProjects2/dacon_img/img_data')
    parser.add_argument('--save_path', type=str, default='./weights_train')
    parser.add_argument('--pretrain', type=bool, default=True)
    args = parser.parse_args()


    rmse = lambda x, y: torch.sqrt(torch.mean((y - x) ** 2))
    psnr = lambda x, y: 20 * torch.log10(255.0 / rmse(x, y))

    save_name = 'model_unet_best.pt' if args.pretrain else 'model_upp_best.pt'
    os.makedirs(args.save_path, exist_ok=True)

    train_dataset = ImageDataset(args.data_path)
    valid_dataset = ImageDataset(args.data_path, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=13)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                              num_workers=13)
    if not args.pretrain:
        model = smp.UnetPlusPlus(encoder_name=args.model_name, classes=3)
        epoch = 200
    else:
        model = smp.Unet(encoder_name=args.model_name, encoder_weights='imagenet', classes=3)
        epoch = 150

    model = nn.DataParallel(model, device_ids=[0])
    model.cuda()

    if not args.pretrain:
        model.module.encoder.load_state_dict(torch.load(os.path.join(args.save_path, 'model_unet_best.pt')))
        print('Pretrained model weights loaded..')

    scaler = amp.GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    decay_steps = (len(train_dataset) // args.batch_size + 1) * epoch
    plr = PolynomialLRDecay(optimizer, max_decay_steps=decay_steps, end_learning_rate=1e-6, power=0.9)
    criterion_l1 = nn.L1Loss()

    best_psnr = -1
    print('Training Start..', args)
    for epoch in range(epoch):
        avg_l1 = AverageMeter()
        avg_total = AverageMeter()
        t_stt = timer()
        model.train()

        for data in train_loader:
            optimizer.zero_grad()
            input, output, _, _, filename = data
            input = input.cuda()
            output = output.cuda()

            with amp.autocast(): # We used mixed-precision to boost the learning speed.
                model_prediction = model(input) # Our model have to generate residual between input and output image.
                prediction = torch.clamp((input + model_prediction), 0., 1.) # The generated output should be 0-1
                l1_loss = criterion_l1(prediction, output) + criterion_l1(model_prediction, (output-input).detach())
                # The first term is used to minimize the distance between prediction and output, the second term is minimized to learn "Residual"
                total_loss = l1_loss

            avg_l1.update(l1_loss.item(), input.shape[0])
            avg_total.update(total_loss.item(), input.shape[0])
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            plr.step()
            
        t_end = timer()
        v_stt = timer()
        model.eval()
        
        with torch.no_grad():
            avg_val_l1 = AverageMeter()
            avg_val_total = AverageMeter()
            avg_val_psnr = AverageMeter()
            for data in  valid_loader:
                input, output, _, _, filename = data
                input = input.cuda()
                output = output.cuda()

                with amp.autocast():
                    model_prediction = model(input)
                    prediction = torch.clamp((input + model_prediction), 0., 1.)
                    l1_loss = criterion_l1(prediction, output)
                    total_loss = l1_loss

                avg_val_l1.update(l1_loss.item(), input.shape[0])
                avg_val_total.update(total_loss.item(), input.shape[0])

                prediction = torch.clamp(prediction * 255.0, 0., 255.)
                input = torch.clamp(input * 255.0, 0., 255.)
                output = torch.clamp(output * 255.0, 0., 255.)

                for origin_x, origin_y, gen_y in zip(input, output, prediction):
                    avg_val_psnr.update(psnr(gen_y, origin_y).item(), 1)
                
        v_end = timer()

        print("[Epoch %03d] Train loss[Total: %.6f, l1: %.6f]" % (epoch, avg_total.avg, avg_l1.avg))
        print("[Epoch %03d] Valid loss[Total: %.6f l1: %.6f psnr: %.6f]" % (
              epoch, avg_val_total.avg, avg_val_l1.avg, avg_val_psnr.avg))
        print("[Elapsed Time] train", timedelta(seconds=t_end - t_stt), 'valid', timedelta(seconds=v_end - v_stt))
        print("---")


        if best_psnr < avg_val_psnr.avg:
            print("PSNR has been improved.", best_psnr, '-->', avg_val_psnr.avg)
            torch.save(model.module.encoder.state_dict(), os.path.join(args.save_path, save_name))
            best_psnr = avg_val_psnr.avg
            
        

if __name__ == "__main__":
    main()
