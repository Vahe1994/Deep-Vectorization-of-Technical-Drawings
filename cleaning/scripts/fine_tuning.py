import argparse
import os
from time import gmtime, strftime

import numpy as np
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F

from cleaning.utils.dataloader import MakeDataSynt
from cleaning.utils.loss import IOU, CleaningLoss
from util_files.metrics.raster_metrics import iou_score
from cleaning.models.Unet.unet_model import UNet
from cleaning.models.SmallUnet.unet import SmallUnet


MODEL_LOSS = {
    'UNET': {
        'model': UNet(n_channels=3,n_classes=1,final_tanh=True),
        'loss': CleaningLoss(kind='BCE', with_restore=False)
    },
    'SmallUNET': {
        'model': SmallUnet(),
        'loss': CleaningLoss(kind='BCE', with_restore=False)
    },
    'UNET_MSE': {
        'model': UNet(n_channels=3,n_classes=1),
        'loss': CleaningLoss(kind='MSE', with_restore=False)
    },

}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True, help='What model to use, one of [ "UNET,"UNET_MSE,"SmallUnet"]', default='UNET')
    parser.add_argument('--n_epochs', type=int, help='Num of epochs for training', default=10)
    parser.add_argument('--datadir', type=str, help='Path to training dataset')
    parser.add_argument('--valdatadir', type=str, help='Path to validation dataset')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--name', type=str, help='Name of the experiment')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model checkpoint')

    args = parser.parse_args()
    return args

def get_dataloaders(args):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ColorJitter()
    ])
    #TODO Make sure that this should be MakeDataSynt and not MakeData from dataloader.py
    dset_synt = MakeDataSynt(args.datadir, args.datadir, train_transform, 1)
    dset_val_synt = MakeDataSynt(args.valdatadir, args.valdatadir, train_transform)

    print(args.batch_size)

    train_loader = DataLoader(dset_synt,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=0,  # 1 for CUDA
                              pin_memory=False)

    val_loader = DataLoader(dset_val_synt,
                            batch_size=1,
                            shuffle=True,
                            num_workers=1,  # 1 for CUDA
                            pin_memory=False)

    return train_loader, val_loader


def validate(tb, val_loader, model, loss_func, global_step):
    val_loss_epoch = []
    val_iou_extract = []

    for x_input, y_extract, y_restor in tqdm(val_loader):
        x_input = torch.FloatTensor(x_input).cuda()
        y_extract = y_extract.type(torch.FloatTensor).cuda()

        logits_restor, logits_extract = None, model(x_input)  # restoration + extraction

        if args.added_part == "Yes":
            # training "Unet on without filling wholes on h_gt in synthetic
            loss = loss_func(logits_extract, logits_restor, y_extract, y_restor)
            iou_scr = iou_score(torch.round(torch.clamp(logits_extract, 0, 1).cpu()).long().numpy(),
                                y_extract.cpu().long().numpy())
        else:
            # training "Unet on whole image nh_gt, default this one
            loss = loss_func(logits_extract, logits_restor, y_restor, y_extract)
            iou_scr = iou_score(torch.round(torch.clamp(logits_extract, 0, 1).cpu()).long().numpy(),
                                y_restor.cpu().long().numpy())

        val_iou_extract.append(iou_scr)
        val_loss_epoch.append(loss.cpu().data.numpy())
        del loss

    tb.add_scalar('val_loss', np.mean(val_loss_epoch), global_step=global_step)

    tb.add_scalar('val_iou_extract', np.mean(val_iou_extract), global_step=global_step)
    out_grid = torchvision.utils.make_grid(logits_extract.unsqueeze(1).cpu())
    input_grid = torchvision.utils.make_grid(x_input.cpu())
    tb.add_image(tag='val_out_extract', img_tensor=out_grid, global_step=global_step)
    tb.add_image(tag='val_input', img_tensor=input_grid, global_step=global_step)

def save_model(model, path):
    print('Saving model to "%s"' % path)
    torch.save(model, path)


def load_model(model, path):
    print('Loading model from "%s"' % path)
    model = torch.load(path)

    return model


def main(args):
    tb_dir = '/logs/tb_logs_article/fine_tuning_' + args.name 
    tb = SummaryWriter(tb_dir)

    train_loader, val_loader = get_dataloaders(args)

    if args.model not in ["UNET", 'UNET_MSE', 'SmallUnet']:
        raise Exception('Unsupported type of model, choose from ["UNET,"UNET_MSE,"SmallUnet"]')

    model = MODEL_LOSS[args.model]['model']
    loss_func = MODEL_LOSS[args.model]['loss']

    model = model.cuda()

    if 'model_path' in args and args.model_path is not None:
        model = load_model(model, args.model_path)
    model.eval()
    tb.add_text(tag='model', text_string=repr(model))

    opt = torch.optim.Adam(model.parameters(), lr=0.0005)

    global_step = 0

    for epoch in range(args.n_epochs):
        model.train()
        for x_input, y_extract, y_restor in tqdm(train_loader):
            x_input = torch.FloatTensor(x_input).cuda()
            y_extract = y_extract.type(torch.FloatTensor).cuda()

            logits_restor, logits_extract = None, model(x_input)  # restoration + extraction
            #training "Unet on whole image, aka the final result, default this one, make sure this image is in y_restor
            #or swap  y_extract,y_restor
            loss = loss_func(logits_extract, logits_restor, y_restor, y_extract)
            loss.backward()
            opt.step()
            opt.zero_grad()

            tb.add_scalar('train_loss', loss.cpu().data.numpy(), global_step=global_step)

            if global_step % 500 == 0:
                out_grid = torchvision.utils.make_grid(logits_extract.unsqueeze(1).cpu())
                input_grid = torchvision.utils.make_grid(x_input.cpu())

                tb.add_image(tag='train_out_extract', img_tensor=out_grid, global_step=global_step)
                tb.add_image(tag='train_input', img_tensor=input_grid, global_step=global_step)

                model.eval()
                with torch.no_grad():
                    validate(tb, val_loader, model, loss_func, global_step=global_step)
                model.train()
                save_model(model, os.path.join(tb_dir, 'model_it_%s.pth' % global_step))

            del logits_extract
            del logits_restor

            global_step += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)



