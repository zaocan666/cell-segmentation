import os
import torch
import cv2
import numpy as np
import imageio
import matplotlib.pyplot as plt
import re
from torchvision.utils import save_image

from utils.dataset import preprocess_img_gt, crop_tile, all_transform, CROP_FLAG
from unet import UNet
from instance import instancing

def visual(gt):
    label = np.unique(gt)
    height, width = gt.shape[:2]
    visual_img = np.zeros((height, width, 3))
    for lab in label:
        if lab == 0:
            continue
        color = np.random.randint(low=0, high=255, size=3)
        visual_img[gt==lab, :] = color
    return visual_img.astype(np.uint8)

def load_images(file_names):
    images = []
    for file_name in file_names:
        img = cv2.imread(file_name, -1)
        # img = img_standardization(img)
        images.append(img)
    return images

def segmentor(net, img, device):
    img[img>255]=255
    # img = np.array(img, dtype=np.float)
    img = np.array(img, dtype=np.uint8)

    img = np.repeat(img[:,:,None],3,axis=2)
    img = all_transform(img)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    # img=img.repeat([3,1,1])

    with torch.no_grad():
        if CROP_FLAG:
            output=crop_tile(net, img)
        else:
            output = net(img)
        probs = torch.sigmoid(output)

        full_mask = probs.squeeze(0).squeeze(0)
        bi_mask=(full_mask > 0.5).int()

        pred_mask = instancing(bi_mask.cpu().numpy())
    return pred_mask, img, bi_mask.cpu().numpy()

if __name__ == '__main__':
    root_path='checkpoints'
    model_path=[os.path.join(root_path, mod) for mod in os.listdir(root_path) if mod.find('.pth') != -1]
    model_path={int(re.findall(r"\d+",x)[0]): x for x in model_path}
    model_path=model_path[2] #196

    image_path = r'/home/kaixi_whli/segment/supplementary_modify/dataset1/test'
    result_path = 'test_RES'
    vis_path = 'test_RES_vis'

    if not os.path.exists(result_path):
        os.mkdir(result_path)
    if not os.path.exists(vis_path):
        os.mkdir(vis_path)

    image_list = sorted([os.path.join(image_path, image) for image in os.listdir(image_path)])

    device = torch.device('cuda')
    net = UNet(n_channels=3, n_classes=1)
    net = torch.nn.DataParallel(net)
    # logging.info("Loading model {}".format(model_vali))        
    net.to(device=device)
    net.load_state_dict(torch.load(model_path, map_location=device))

    images = load_images(image_list)
    for index, image in enumerate(images):
        label_img, input_img, bi_mask = segmentor(net, image, device)
        label_vis = visual(label_img)
        imageio.imwrite(os.path.join(result_path, 'mask{:0>3d}.tif'.format(index)), label_img.astype(np.uint16))

        input_img = input_img.cpu().numpy()+1
        label_vis = np.transpose(label_vis, [2,0,1])
        imageio.imwrite(os.path.join(result_path, 'mask{:0>3d}.tif'.format(index)), label_img.astype(np.uint16))
        comb_img = torch.Tensor([input_img.squeeze(0), label_vis/255.0])
        save_image(comb_img, os.path.join(vis_path, 'mask{:0>3d}_0.png'.format(index)))
        # imageio.imwrite(os.path.join(vis_path, 'mask{:0>3d}_2bi_vis.png'.format(index)), (bi_mask * 255).astype(np.uint8))

        # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 8))
        # ax0, ax1 = axes
        # ax0.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
        # ax1.imshow(label_img, cmap=plt.cm.Spectral, interpolation='nearest')
        # plt.show()

