import os
import torch
import cv2
import numpy as np
import imageio
import matplotlib.pyplot as plt
import re
import argparse

from utils.dataset import preprocess_img_gt, crop_tile, all_transform, CROP_FLAG
from unet import UNet
from instance import instancing
from predict import visual

def load_images(file_names):
    images = []
    for file_name in file_names:
        img = cv2.imread(file_name, -1)
        # img = img_standardization(img)
        images.append(img)
    return images

def segmentor(net, img, device):
    img, _, _ = preprocess_img_gt(img, np.zeros(img.shape), np.zeros(img.shape), False)
    # img[img>255]=255
    # img = np.array(img, dtype=np.float)
    # img = np.array(img, dtype=np.uint8)

    # img = np.repeat(img[:,:,None],3,axis=2)
    # img = all_transform(img)

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    # img=img.repeat([3,1,1])

    net.eval()
    with torch.no_grad():
        if CROP_FLAG:
            output=crop_tile(net, img)
        else:
            output = net(img)
        probs = torch.sigmoid(output)

        full_mask = probs.squeeze(0)
        bi_full_mask=(full_mask > 0.5).int()
        bi_mask = bi_full_mask[0]
        bi_contour = bi_full_mask[1]

        pred_mask = instancing(bi_mask.cpu().numpy(), bi_contour.cpu().numpy(), img[0,0].cpu().numpy())
    return pred_mask, bi_mask.cpu().numpy(), bi_contour.cpu().numpy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-image', '--image_path', metavar='B', type=str, nargs='?', help='path of imgs')
    args = parser.parse_args()

    root_path='checkpoints'
    model_path=[os.path.join(root_path, mod) for mod in os.listdir(root_path) if mod.find('.pth') != -1]
    model_path={int(re.findall(r"\d+",x)[1]): x for x in model_path}
    model_path=model_path[115-1] #196
    # image_path = r'/home/kaixi_whli/segment/supplementary_modify/dataset2/test'
    image_path = args.image_path
    result_path = 'test_RES'
    vis_path = 'test_RES_vis'

    if not os.path.exists(result_path):
        os.mkdir(result_path)
    if not os.path.exists(vis_path):
        os.mkdir(vis_path)

    image_list = sorted([os.path.join(image_path, image) for image in os.listdir(image_path) if image.find('.tif') ==-1])

    device = torch.device('cuda')
    net = UNet(n_channels=3, n_classes=2)
    net = torch.nn.DataParallel(net)
    # logging.info("Loading model {}".format(model_vali)) 
    print('model path:', model_path)       
    net.to(device=device)
    net.load_state_dict(torch.load(model_path, map_location=device))

    images = load_images(image_list)
    for index, image in enumerate(images):
        label_img, bi_mask, bi_contour = segmentor(net, image, device)
        input_image, label_vis = visual(image, label_img)
        imageio.imwrite(os.path.join(result_path, 'mask{:0>3d}.tif'.format(index)), label_img.astype(np.uint16))
        imageio.imwrite(os.path.join(vis_path, 'mask{:0>3d}_0.png'.format(index)), input_image.astype(np.uint8))
        imageio.imwrite(os.path.join(vis_path, 'mask{:0>3d}_1label.png'.format(index)), label_vis.astype(np.uint8))
        imageio.imwrite(os.path.join(vis_path, 'mask{:0>3d}_2bi_vis.png'.format(index)), (bi_mask * 255).astype(np.uint8))
        imageio.imwrite(os.path.join(vis_path, 'mask{:0>3d}_3bi_contour_vis.png'.format(index)), (bi_contour * 255).astype(np.uint8))

        # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 8))
        # ax0, ax1 = axes
        # ax0.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
        # ax1.imshow(label_img, cmap=plt.cm.Spectral, interpolation='nearest')
        # plt.show()

