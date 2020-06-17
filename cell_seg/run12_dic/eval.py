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

    score_num = 0
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
                tot += dice_coeff(pred_mask, true_biMasks).item() * pred_mask.shape[0]
                tot_contour  += dice_coeff(pred_contour, true_biMasks).item() * pred_mask.shape[0]
                score_num += pred_mask.shape[0]
                # tot += nn.BCEWithLogitsLoss(reduction='mean')(mask_pred, true_biMasks).item()
            pbar.update()
            
    net.train()
    return tot / score_num, tot_contour/score_num

# def jaccard(pred, true_mask):
#     #pred [h, w] true_mask [h, w]
#     pred_index = np.unique(np.reshape(pred, -1))
#     pred_index = pred_index[pred_index!=0]
#     pred_areas = [np.array(pred==i, dtype=np.int) for i in pred_index] #[n, h, w] 1,0
#     pred_areas = np.array(pred_areas)

#     pred_num = pred_areas.shape[0]
#     pred_areas = np.reshape(pred_areas, (pred_num, -1)).astype(np.int32)  # [n, h*w]
#     pred_sum = np.sum(pred_areas, axis=1)  # [n]

#     mask_index = np.unique(np.reshape(true_mask, -1))
#     mask_index = mask_index[mask_index != 0]

#     jaccard_score = 0
#     for mask_i in mask_index:
#         maski_area = np.array(true_mask==mask_i, dtype=np.int) #[h, w] 1.0

#         maski_area = np.reshape(maski_area, (-1)).astype(np.int32) #[h*w]
#         maski_sum = np.sum(maski_area)  # x

#         # intersections and union
#         intersections = np.sum(maski_area[None, :] * pred_areas, axis=1)  # [n]
#         union = maski_sum + pred_sum - intersections
#         score_i = intersections / (union + 1e-6)

#         if score_i.max()>0.5:
#             jaccard_score += score_i.max()

#     div_low = max([len(pred_index), len(mask_index)])
#     return jaccard_score/div_low

def cross_compute_mask_iou(gt_mask, pred_mask):
    """Computes IoU overlaps between one and a batch of masks. Regard different classes as the same.
    gt_mask: [H, W]
    pred_masks: [B, H, W]
    Returns ious of the two masks.
    """
    # flatten masks and compute their areas
    batch = pred_mask.shape[0]
    gt_mask[gt_mask > 0] = 1
    pred_mask[pred_mask > 0] = 1
    gt_mask = np.reshape(gt_mask, (-1)).astype(np.int32)  # [K]
    pred_mask = np.reshape(pred_mask, (batch, -1)).astype(np.int32)  # [B, K]
    area1 = np.sum(gt_mask)  # a scalar
    area2 = np.sum(pred_mask, axis=1)  # [B]
    # intersections and union
    intersections = np.sum(gt_mask[None, :] * pred_mask, axis=1)  # [B]
    union = area1 + area2 - intersections
    ious = intersections / (union + 1e-12)  # avoid intersections to be divided by 0
    return ious


def jaccard(gt_mask, pred_mask):
    """This is the class aware implementation of Jaccard score.
    Find max overlap > 0.5 region (just one connected region) in pred_mask
        corresponding to every gt cell.
    gt_mask, pred_mask: [H, W]
    Final score average over predict cell number.
    """
    gt_cells_idx = np.unique(gt_mask)
    gt_cells_idx = gt_cells_idx[gt_cells_idx != 0]
    pred_cells_idx = np.unique(pred_mask)
    pred_cells_idx = pred_cells_idx[pred_cells_idx != 0]
    pred_cell_regions = [(pred_mask == pred_idx).astype(np.int) for pred_idx in pred_cells_idx]
    pred_cell_regions = np.array(pred_cell_regions)
    iou_sum = 0.
    for gt_idx in gt_cells_idx:
        gt_cell_region = (gt_mask == gt_idx).astype(np.int)
        cross_iou = cross_compute_mask_iou(gt_cell_region, pred_cell_regions)  # [num_pred_cells]
        if cross_iou.max() <= 0.5:
            # TODO: no punishment here
            continue
        iou_sum += cross_iou.max()
    # num = max(len(gt_cells_idx), len(pred_cells_idx))
    num = len(gt_cells_idx)
    return iou_sum / float(num)