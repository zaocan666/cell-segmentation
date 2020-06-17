import os
import torch
import cv2
import numpy as np
import imageio
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import argparse

from utils.dataset import preprocess_img_gt, crop_tile, all_transform
from unet32 import UNet as UNet32
from Unet_change import AttU_Net
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

def segmentor(models, img, device, crop_flag, vis_flag=False):

    img[img>255]=255
    # img = np.array(img, dtype=np.float)
    img = np.array(img, dtype=np.uint8)

    img = np.repeat(img[:,:,None],3,axis=2)
    img = all_transform(img)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    # img=img.repeat([3,1,1])

    with torch.no_grad():
        # output = net(img)
        full_mask = torch.zeros([img.shape[2], img.shape[3]]).cuda()
        for i, net in enumerate(models):
            net.eval()
            if crop_flag[i]:
                output=crop_tile(net, img)
            else:
                output = net(img)

            probs = torch.sigmoid(output)

            full_mask += probs.squeeze(0).squeeze(0)
        full_mask /= len(models)
        bi_mask=(full_mask > 0.5).int()

        pred_mask = instancing(bi_mask.cpu().numpy(), vis_flag=vis_flag)
    return pred_mask, img, bi_mask.cpu().numpy()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
	                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-path', '--model_base_path', metavar='E', type=str, help='path of models')
	parser.add_argument('-image', '--image_path', metavar='B', type=str, nargs='?', help='path of imgs')
	args = parser.parse_args()
	# model_base_path='/home/kaixi_whli/segment/data1/run46_40/checkpoints'
	model_base_path=args.model_base_path

	model_paths = [model for model in os.listdir(model_base_path) if model.find('.pth') != -1 and model.find('best') != -1]
	model_class = ['unet32' for i in range(len(model_paths))]
	crop_flag = [True for i in range(len(model_paths))]

	# image_path = r'/home/kaixi_whli/segment/supplementary_modify/dataset1/test'
	image_path = args.image_path
	result_path = 'test_RES'
	vis_path = 'test_RES_vis'

	if not os.path.exists(result_path):
	    os.mkdir(result_path)
	if not os.path.exists(vis_path):
	    os.mkdir(vis_path)

	image_list = sorted([os.path.join(image_path, image) for image in os.listdir(image_path)])

	device = torch.device('cuda')
	models = []
	print('start')
	for i in range(len(model_paths)):
	    if model_class[i]=='unet32':
	        net = UNet32(n_channels=3, n_classes=1)
	    elif model_class[i]=='atten':
	        net = AttU_Net()

	    net = torch.nn.DataParallel(net)
	    net.to(device=device)
	    model_p = os.path.join(model_base_path, model_paths[i])
	    print('loading:', model_paths[i])
	    net.load_state_dict(torch.load(model_p, map_location=device))
	    models.append(net)

	images = load_images(image_list)
	for index, image in enumerate(images):
	    label_img, input_img, bi_mask = segmentor(models, image, device, crop_flag, index==-1)
	    label_vis = visual(label_img)

	    input_img = input_img.cpu().numpy()+1
	    label_vis = np.transpose(label_vis, [2,0,1])
	    imageio.imwrite(os.path.join(result_path, 'mask{:0>3d}.tif'.format(index)), label_img.astype(np.uint16))
	    # imageio.imwrite(os.path.join(vis_path, 'mask{:0>3d}_0.png'.format(index)), input_image.astype(np.uint8))
	    comb_img = torch.Tensor([input_img.squeeze(0), label_vis/255.0])
	    save_image(comb_img, os.path.join(vis_path, 'mask{:0>3d}_0.png'.format(index)))
	    # imageio.imwrite(os.path.join(vis_path, 'mask{:0>3d}_1label.png'.format(index)), label_vis.astype(np.uint8))
	    # imageio.imwrite(os.path.join(vis_path, 'mask{:0>3d}_2bi_vis.png'.format(index)), (bi_mask * 255).astype(np.uint8))

	    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 8))
	    # ax0, ax1 = axes
	    # ax0.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
	    # ax1.imshow(label_img, cmap=plt.cm.Spectral, interpolation='nearest')
	    # plt.show()

