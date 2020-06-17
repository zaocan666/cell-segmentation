import argparse
import logging
import os
import sys
import numpy as np
import random

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet
from dice_loss import dice_coeff
from predict import predict_netOnDataList

from tensorboardX import SummaryWriter
from utils.dataset import BasicDataset, make_dataset
from torch.utils.data import DataLoader

dataset_dir = r'utils/data_list.txt'
# root_dir = r'/home/kaixi_whli/segment/supplementary_modify/dataset2'
root_dir = ' '
dir_checkpoint = 'checkpoints/'
n_classes=1
n_channels=3

def train_net(net,
              device,
              writer_detail,
              writer_main,
              random_seed,
              epochs=5,
              batch_size=1,
              lr=0.001,
              save_cp=True,
              val_i=4,
              ):

    train_dataset, val_dataset, train_list, val_list = make_dataset(root_dir, dataset_dir, val_i)
    n_val = val_dataset.__len__()
    n_train = train_dataset.__len__()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True, worker_init_fn=np.random.seed(random_seed))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False, worker_init_fn=np.random.seed(random_seed))

    global_step = 0

    logging.info(f'''Starting training:
        val_i:           {val_i}
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')

    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adam(params=net.parameters(), lr=lr)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if n_classes > 1 else 'max', patience=2)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=np.power(0.1, 1/epochs))

    if n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
    

    best_score=0
    best_net=net
    best_e=0
    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                contour_masks = batch['mask_contour']
                true_biMasks = (true_masks>0).int()
                contour_masks = (contour_masks>0).int()
                assert imgs.shape[1] == n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if n_classes == 1 else torch.long
                true_biMasks = true_biMasks.to(device=device, dtype=mask_type)
                contour_masks = contour_masks.to(device=device, dtype=mask_type)
                # combine_masks = torch.cat([true_biMasks, contour_masks], dim=1) #

                masks_pred = net(imgs)
                loss_mask= criterion(masks_pred[:,0:1], true_biMasks)
                loss_contour = criterion(masks_pred[:,1:2], contour_masks)
                loss = loss_mask + 10*loss_contour
                epoch_loss += loss.item()
                writer_main.add_scalar('val_%d_Loss/train'%val_i, loss.item(), global_step)

                pbar.set_postfix(loss_mask = loss_mask.item(), loss_contour=loss_contour.item())

                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % ((n_train+n_val) // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer_detail.add_histogram('val_%d_weights/'%val_i + tag, value.data.cpu().numpy(), global_step)
                        writer_detail.add_histogram('val_%d_grads/'%val_i + tag, value.grad.data.cpu().numpy(), global_step)

                    writer_detail.add_images('val_%d_images'%val_i, imgs[0:1], global_step)
                    if n_classes == 1:
                        writer_detail.add_images('val_%d_masks/true'%val_i, true_biMasks[0:1], global_step)
                        writer_detail.add_images('val_%d_masks/pred'%val_i, torch.sigmoid(masks_pred[0:1,0:1]) > 0.5, global_step)
                        writer_detail.add_images('val_%d_masks_contour/true'%val_i, contour_masks[0:1], global_step)
                        writer_detail.add_images('val_%d_masks_contour/pred'%val_i, torch.sigmoid(masks_pred[0:1,1:2]) > 0.5, global_step)

        val_score_mask, val_score_contour = eval_net(net, val_loader, device, n_classes)
        # scheduler.step(val_score)
        writer_main.add_scalar('val_%d_learning_rate'%val_i, optimizer.param_groups[0]['lr'], global_step)

        # if val_score>=best_score:
        #     best_score=val_score
        #     best_net=net
        #     best_e=epoch

        logging.info('val {} Validation score mask: {} Validation score contour: {}'.format(val_i, val_score_mask, val_score_contour))
        # if val_score<10:
        writer_main.add_scalar('val_%d_score_mask/test'%val_i, val_score_mask, global_step)
        writer_main.add_scalar('val_%d_score_contour/test'%val_i, val_score_contour, global_step)

        scheduler.step()

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'val {val_i}_CP_epoch{epoch + 1}_scoreMask_%.4f_scoreContour_%.4f.pth'%(val_score_mask,val_score_contour))
        # logging.info(f'val {val_i} Checkpoint {epoch + 1} saved, best score: %.4f !'%best_score)

    # result_path='result_img'
    # if not os.path.exists(result_path):
    #     os.mkdir(result_path)
    # jaccard_vali = predict_netOnDataList(root_dir, best_net, val_list, device, result_path=result_path, vali=val_i, mask_threhold=0.5, vis_flag=True)
    # logging.info('val %d epoch %d jaccard score: %.6f'%(val_i, best_e+1, jaccard_vali))


    # if save_cp:
    #     try:
    #         os.mkdir(dir_checkpoint)
    #         logging.info('Created checkpoint directory')
    #     except OSError:
    #         pass
    #     torch.save(best_net.state_dict(),
    #                dir_checkpoint + f'val {val_i}_CP_epoch{best_e + 1}_score_%.4f_jaccard_%.4f.pth'%(best_score,jaccard_vali))
    #     logging.info(f'val {val_i} Checkpoint {best_e + 1} saved, best score: %.4f !'%best_score)

    # return jaccard_vali
    return 0


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=200,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=15,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.01,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=20.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-random_seed', '--random_seed', type=int, default=5123,
                        help='random seed')
    parser.add_argument('-root_dir', '--root_dir', type=str,
                        help='dataset root dir')

    return parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    torch.set_num_threads(1)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    root_dir = args.root_dir
    setup_seed(args.random_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    writer_main = SummaryWriter(comment=f'_main_LR_{args.lr}_BS_{args.batchsize}')
    writer_detail = SummaryWriter(comment=f'_detail_LR_{args.lr}_BS_{args.batchsize}')

    best_scores=[]
    for val_i in [0]:
        net = UNet(n_channels=n_channels, n_classes=2, bilinear=True)
        logging.info(f'val {val_i} Network:\n'
                     f'\t{net.n_channels} input channels\n'
                     f'\t{net.n_classes} output channels (classes)\n'
                     f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

        if args.load:
            net.load_state_dict(
                torch.load(args.load, map_location=device)
            )
            logging.info(f'Model loaded from {args.load}')

        net.to(device=device)
        net = nn.DataParallel(net)

        # faster convolutions, but more memory
        # cudnn.benchmark = True

        try:
            best_score_i=train_net(net=net,
                      epochs=args.epochs,
                      batch_size=args.batchsize,
                      lr=args.lr,
                      device=device,
                      random_seed=args.random_seed,
                      val_i=val_i,
                      writer_main=writer_main,
                      writer_detail=writer_detail,)

            best_scores.append(best_score_i)

        except KeyboardInterrupt:
            # torch.save(net.state_dict(), 'INTERRUPTED.pth')
            # logging.info('Saved interrupt')
            writer_main.close()
            writer_detail.close()
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)

    writer_main.close()
    writer_detail.close()
    print('best_socres:', best_scores)
    print('average:', sum(best_scores)/len(best_scores))
