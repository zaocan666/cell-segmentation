import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import torch.nn as nn

from dice_loss import dice_coeff
from utils.dataset import crop_tile, CROP_FLAG

def eval_net(net, loader, device, n_classes):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    tot_contour = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            true_biMasks = (true_masks>0).int()
            contour_masks = batch['mask_contour']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_biMasks = true_biMasks.to(device=device, dtype=mask_type)
            contour_masks = contour_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                if CROP_FLAG:
                    mask_pred=crop_tile(net, imgs)
                else:
                    mask_pred = net(imgs)

            if n_classes > 1:
                tot += F.cross_entropy(mask_pred, true_biMasks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                pred_mask = pred[:,0]
                pred_contour = pred[:,1]
                tot += dice_coeff(pred_mask, true_biMasks).item()
                tot_contour  += dice_coeff(pred_contour, true_biMasks).item()
                # tot += nn.BCEWithLogitsLoss(reduction='mean')(mask_pred, true_biMasks).item()
            pbar.update()
            
    net.train()
    return tot / n_val, tot_contour/n_val

def jaccard(pred, true_mask):
    #pred [h, w] true_mask [h, w]
    pred_index = np.unique(np.reshape(pred, -1))
    pred_index = pred_index[pred_index!=0]
    pred_areas = [np.array(pred==i, dtype=np.int) for i in pred_index] #[n, h, w] 1,0
    pred_areas = np.array(pred_areas)

    pred_num = pred_areas.shape[0]
    pred_areas = np.reshape(pred_areas, (pred_num, -1)).astype(np.int32)  # [n, h*w]
    pred_sum = np.sum(pred_areas, axis=1)  # [n]

    mask_index = np.unique(np.reshape(true_mask, -1))
    mask_index = mask_index[mask_index != 0]

    jaccard_score = 0
    for mask_i in mask_index:
        maski_area = np.array(true_mask==mask_i, dtype=np.int) #[h, w] 1.0

        maski_area = np.reshape(maski_area, (-1)).astype(np.int32) #[h*w]
        maski_sum = np.sum(maski_area)  # x

        # intersections and union
        intersections = np.sum(maski_area[None, :] * pred_areas, axis=1)  # [n]
        union = maski_sum + pred_sum - intersections
        score_i = intersections / (union + 1e-6)

        if score_i.max()>0.5:
            jaccard_score += score_i.max()

    # div_low = max([len(pred_index), len(mask_index)])
    div_low = len(mask_index)
    return jaccard_score/div_low
