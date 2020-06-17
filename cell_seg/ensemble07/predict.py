import argparse
import logging
import os

import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms

from unet import UNet
from utils.dataset import BasicDataset, get_list, preprocess_img_gt, crop_tile
from instance import instancing
from eval import jaccard

VISILE=True

def predict_img(net,
                full_img,
                gt,
                device,
                out_threshold=0.5):
    net.eval()

    img, gt = preprocess_img_gt(full_img, gt, False)
    gt=gt.squeeze(0)

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        # output = net(img)
        output=crop_tile(net, img)
        probs = torch.sigmoid(output)

        full_mask = probs.squeeze(0).squeeze(0)
        bi_mask=(full_mask > out_threshold).int()

        pred_mask = instancing(bi_mask.cpu().numpy())
        result = jaccard(pred_mask, gt.numpy())

    return result, pred_mask, bi_mask.cpu().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)

    return parser.parse_args()

def visual(img, gt):
    label = np.unique(gt)
    height, width = img.shape[:2]
    visual_img = np.zeros((height, width, 3))
    for lab in label:
        if lab == 0:
            continue
        color = np.random.randint(low=0, high=255, size=3)
        visual_img[gt==lab, :] = color
    return img.astype(np.uint8), visual_img.astype(np.uint8)

def predict_netOnDataList(root_dir, net, val_list, device, mask_threhold, vali, result_path, vis_flag=VISILE):
    vali_path = os.path.join(result_path, 'val'+str(vali))
    if VISILE and (not os.path.exists(vali_path)):
        os.mkdir(vali_path)
    jaccard_sum = 0
    for i, fn in enumerate(val_list):
        # logging.info("\nPredicting image {} ...".format(fn))

        img = cv2.imread(os.path.join(root_dir, fn[0]), -1)
        gt = cv2.imread(os.path.join(root_dir, fn[1]), -1)

        jaccard_i, pred_mask, bi_mask = predict_img(net=net,
                                                    full_img=img,
                                                    gt=gt,
                                                    out_threshold=mask_threhold,
                                                    device=device)
        jaccard_sum += jaccard_i

        if vis_flag:
            _, gt_vis = visual(img, gt)
            _, pred_vis = visual(img, pred_mask)
            cv2.imwrite(os.path.join(vali_path, '%d_3bi_vis.jpg' % i), (bi_mask * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(vali_path, '%d_2gt_vis.jpg' % i), gt_vis.astype(np.uint8))
            cv2.imwrite(os.path.join(vali_path, '%d_4pred_vis.jpg' % i), pred_vis.astype(np.uint8))
            cv2.imwrite(os.path.join(vali_path, '%d_1origin_img.jpg' % i), img.astype(np.uint8))

        # if not args.no_save:
        #     out_fn = out_files[i]
        #     result = mask_to_image(mask)
        #     result.save(out_files[i])

        #     logging.info("Mask saved to {}".format(out_files[i]))

        # if args.viz:
        #     logging.info("Visualizing results for image {}, close to continue ...".format(fn))
        #     plot_img_and_mask(img, mask)

    return jaccard_sum/len(val_list)

if __name__ == "__main__":
    root_path = 'checkpoints'
    result_path='result_img'
    model_paths = [[os.path.join(root_path, mod), int(mod[4])] for mod in os.listdir(root_path) if mod.find('.pth') != -1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    if VISILE and (not os.path.exists(result_path)):
        os.mkdir(result_path)

    jaccard_all=[]
    for model_vali in model_paths:
        args = get_args()
        dataset_dir = r'utils/data_list.txt'
        root_dir = r'/home/kaixi_whli/segment/supplementary_modify/dataset1'
        train_list, val_list = get_list(dataset_dir, model_vali[1])
        #out_files = get_output_filenames(args)

        net = UNet(n_channels=3, n_classes=1)
        net = torch.nn.DataParallel(net)
        # logging.info("Loading model {}".format(model_vali))

        
        net.to(device=device)
        net.load_state_dict(torch.load(model_vali[0], map_location=device))

        # logging.info("Model loaded !")

        jaccard_vali=predict_netOnDataList(root_dir, net, val_list, device, result_path=result_path, 
            vali=model_vali[1], mask_threhold=args.mask_threshold, vis_flag=VISILE)

        jaccard_all.append(jaccard_vali)
        print('model:', model_vali[0])
        print('jaccard score:', jaccard_all[-1])

    print('jaccard all average:', sum(jaccard_all)/len(jaccard_all))
