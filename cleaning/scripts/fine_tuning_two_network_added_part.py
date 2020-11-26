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
### Todo check this metric or change it
from util_files.metrics.raster_metrics import iou_score
from cleaning.models.Unet.unet_model import UNet
from cleaning.models.SmallUnet.unet import SmallUnet


MODEL_LOSS = {
    'UNET': {
        'gen': UNet(n_channels=1,n_classes=1,final_tanh=True),
        'unet': UNet(n_channels=3,n_classes=1,final_tanh=True),
        'loss': CleaningLoss(kind='BCE', with_restore=False)
    },
    'UNET_MSE': {
        'gen': UNet(n_channels=1,n_classes=1,final_tanh=False),
        'unet': UNet(n_channels=3,n_classes=1,final_tanh=False),
        'loss': CleaningLoss(kind='MSE', with_restore=False)
    },
    'SmallUnet': {
        'gen': SmallUnet(),
        'unet': SmallUnet(),
        'loss': CleaningLoss(kind='MSE', with_restore=False)
    }
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True, help='What model to use, one of [ "UNET","UNET_MSE"]',
                        default='UNET')
    parser.add_argument('--n_epochs', type=int, help='Num of epochs for training', default=10)
    parser.add_argument('--datadir', type=str, help='Path to training dataset')
    parser.add_argument('--valdatadir', type=str, help='Path to validation dataset')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--name', type=str, help='Name of the experiment')

    parser.add_argument('--gen_path', type=str, default=None, help='Path to gen checkpoint')
    parser.add_argument('--unet_path', type=str, default=None, help='Path to unet checkpoint')

    parser.add_argument('--MSE_comb', type=bool, default=False,
                        help='steps to train discriminator [default: False]')
    parser.add_argument('--discrimin_cond', type=bool,default=False,
                        help='steps to train discriminator [default: False]')

    args = parser.parse_args()
    return args


def get_dataloaders(args):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ColorJitter()
    ])

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


def validate(tb, val_loader, unet, gen, loss_func, global_step ):
    val_iou_extract = []
    val_loss_epoch = []
    val_iou_without_gan = []
    unet.eval()

    for x_input, y_extract, y_restor in tqdm(val_loader):
       
        with torch.no_grad():
            x_input = torch.FloatTensor(x_input).cuda()

            # Cleaning prediction
            logits_restor, logits_extract = None, unet(x_input)
            #1 - Cleaning prediction
            logits_extract = 1 - logits_extract.unsqueeze(1)

            y_restor = 1. - y_restor.type(torch.FloatTensor).cuda().unsqueeze(1)

            #generator prediction based on cleaning prediction
            logits_restore = gen.forward(logits_extract).unsqueeze(1)  # restoration + extraction

          

            loss = loss_func(1 - (logits_extract + logits_restore), None, 1- y_restor,None )
            

            val_loss_epoch.append(loss.cpu().data.numpy())
            iou_scr_without_gan = iou_score(1 - torch.round(logits_extract.squeeze(1)).cpu().long().numpy(),1 - torch.round(y_restor.squeeze(1)).cpu().long().numpy())

            val_iou_without_gan.append(iou_scr_without_gan)

            iou_scr = iou_score(1 - torch.round(torch.clamp(logits_extract + logits_restore, 0, 1).squeeze(1).cpu()).long().numpy(),1 - torch.round(y_restor.squeeze(1)).cpu().long().numpy())

            val_iou_extract.append(iou_scr)

    tb.add_scalar('val_iou_extract', np.mean(val_iou_extract), global_step=global_step)
    tb.add_scalar('val_loss', np.mean(val_loss_epoch), global_step=global_step)
    tb.add_scalar('val_iou_without_gan', np.mean(val_iou_without_gan), global_step=global_step)
    

    out_grid = torchvision.utils.make_grid(1.- torch.clamp(logits_extract + logits_restore, 0, 1).cpu())
    input_grid = torchvision.utils.make_grid(1. - logits_extract.cpu())
    true_grid = torchvision.utils.make_grid(1.- y_restor.cpu())
    input_clean_grid = torchvision.utils.make_grid(x_input.cpu())
                
    tb.add_image(tag='val_first_input', img_tensor=input_clean_grid, global_step=global_step)

    tb.add_image(tag='val_out_extract', img_tensor=out_grid, global_step=global_step)
    tb.add_image(tag='val_input', img_tensor=input_grid, global_step=global_step)
    tb.add_image(tag='val_true', img_tensor=true_grid, global_step=global_step)


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

    if args.model not in ["UNET"]:
        raise Exception('Unsupported type of model, choose from [ "UNET"]')

    gen = MODEL_LOSS[args.model]['gen']
    unet =  MODEL_LOSS[args.model]['unet']
    loss_func = MODEL_LOSS[args.model]['loss']

    gen = gen.cuda()
    unet = unet.cuda()

    if 'gen_path' in args and args.gen_path is not None:
        gen = load_model(gen, args.gen_path)
    if 'unet_path' in args and args.disc_path is not None:
        unet = load_model(unet, args.unet_path)

    tb.add_text(tag='gen', text_string=repr(gen))

    gen_opt = torch.optim.Adamax(gen.parameters(), lr=0.00005)
    unet_opt =  torch.optim.Adamax(unet.parameters(), lr=0.00002)


    global_step = 0

    for epoch in range(args.n_epochs):
        gen.train()
        gen_step = 0
        disc_step = 0
        for x_input, y_extract, y_restor in tqdm(train_loader):
            # data reading
#             unet.train()
            x_input = torch.FloatTensor(x_input).cuda()
            y_extract = y_extract.type(torch.FloatTensor).cuda().unsqueeze(1)
            y_restor = 1. - y_restor.type(torch.FloatTensor).cuda().unsqueeze(1)

            unet.eval()
            with torch.no_grad():
                logits_restor, logits_extract = None, unet(x_input)
                logits_extract = 1 - logits_extract.unsqueeze(1)


            logits_restore = gen.forward(logits_extract).unsqueeze(1)  # restoration + extraction

            # if Cleaning loss use this
            gen_loss = loss_func(1 - (logits_extract + logits_restore), None, 1- y_restor,None )
            # else if with_restore =True use this
            # input_fake = torch.cat((logits_extract + logits_restore,logits_extract),dim = 1)
            #
            # gen_loss =  loss_func(logits_extract, input_fake, y_extract, y_restor)


            gen_opt.zero_grad()
            gen_loss.backward()
            gen_opt.step()

            gen_step += 1
                
            if(np.random.random() <=0.5):
                disc_step+=1
                
            global_step += 1

            if global_step <= 1:
                continue

            tb.add_scalar('gen_vectran_loss',gen_loss.item(), global_step=global_step)
#             tb.add_scalar('train_loss', loss.cpu().data.numpy(), global_step=global_step)


            if global_step % 100 == 0 or global_step <= 2:
                out_grid = torchvision.utils.make_grid(1. - torch.clamp(logits_extract + logits_restore, 0, 1).cpu())
                input_grid = torchvision.utils.make_grid(1. - logits_extract.cpu())
                true_grid = torchvision.utils.make_grid(1. - y_restor.cpu())
                input_clean_grid = torchvision.utils.make_grid(x_input.cpu())
                
                tb.add_image(tag='train_first_input', img_tensor=input_clean_grid, global_step=global_step)
                tb.add_image(tag='train_out_extract', img_tensor=out_grid, global_step=global_step)
                tb.add_image(tag='train_input', img_tensor=input_grid, global_step=global_step)
                tb.add_image(tag='train_true', img_tensor=true_grid, global_step=global_step)
                gen.eval()

                with torch.no_grad():
                    unet.eval()
                    validate(tb, val_loader, unet, gen, loss_func, global_step=global_step)

                gen.train()
                unet.train()

                save_model(gen, os.path.join(tb_dir, 'gen_it_%s.pth' % global_step))
                save_model(unet, os.path.join(tb_dir, 'unet_it_%s.pth' % global_step))



if __name__ == '__main__':
    args = parse_args()
    main(args)



