import os
import numpy as np
import cv2
from torch.utils.data import Dataset
import logging
import torchvision.transforms as transforms
from skimage import transform
import torch
import random

CROP_FLAG=True
CROP_SIZE=500
SCALES=[0.8, 1.2]
ROTATION=15

all_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

def get_list(dataset_dir, val_i):
    train_list=[]
    val_list=[]
    with open(dataset_dir, 'r') as f:
        lines=f.readlines()
        for l in lines:
            strs = l.strip().split(' ')
            val_i_true=int(strs[3])
            if val_i_true==val_i:
                val_list.append([strs[0], strs[1], strs[2]])
            else:
                train_list.append([strs[0], strs[1], strs[2]])
    return train_list, val_list

def rand_crop(data,label,contour):
    height1 = random.randint(0, data.shape[0] - CROP_SIZE)
    width1 = random.randint(0, data.shape[1] - CROP_SIZE )
    height2 = height1 + CROP_SIZE
    width2 = width1 + CROP_SIZE
            
    data=data[height1:height2, width1:width2]
    label=label[height1:height2, width1:width2]
    contour=contour[height1:height2, width1:width2]
 
    return data,label,contour

def rand_resize(data, label):
    scale_act = SCALES[0] + (SCALES[1]-SCALES[0])*np.random.random()
    # data = cv2.resize(data, dsize=(0, 0), fx=scale_act, fy=scale_act)
    # label = cv2.resize(label, dsize=(0, 0), fx=scale_act, fy=scale_act)
    data = transform.resize(data,[round(data.shape[0]*scale_act),round(data.shape[1]*scale_act)], mode="constant", clip=False,preserve_range=True)
    label = transform.resize(label,[round(label.shape[0]*scale_act),round(label.shape[1]*scale_act)], mode="constant", clip=False,preserve_range=True)    
    return np.round(data), np.array(label>0, np.uint8)

def random_flip(data, label):
    if np.random.random()>0.5:
        data=cv2.flip(data, 1)
        label=cv2.flip(label, 1)
    if np.random.random()>0.5:
        data=cv2.flip(data, 0)
        label=cv2.flip(label, 0)
    return data, label

def random_rotation(crop_img, crop_seg, rich_crop_max_rotation, mean_value):
    """
    ??????????

    Args?
        crop_img(numpy.ndarray): ????
        crop_seg(numpy.ndarray): ???
        rich_crop_max_rotation(int)????????0-90
        mean_value(list)???, ??????????????????

    Returns?
        ??????????

    """
    ignore_index = 0
    if rich_crop_max_rotation > 0:
        (h, w) = crop_img.shape[:2]
        do_rotation = np.random.uniform(-rich_crop_max_rotation,
                                        rich_crop_max_rotation)
        pc = (w // 2, h // 2)
        r = cv2.getRotationMatrix2D(pc, do_rotation, 1.0)
        cos = np.abs(r[0, 0])
        sin = np.abs(r[0, 1])

        nw = int((h * sin) + (w * cos))
        nh = int((h * cos) + (w * sin))

        (cx, cy) = pc
        r[0, 2] += (nw / 2) - cx
        r[1, 2] += (nh / 2) - cy
        dsize = (nw, nh)
        crop_img = cv2.warpAffine(
            crop_img,
            r,
            dsize=dsize,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=mean_value)
        crop_seg = cv2.warpAffine(
            crop_seg,
            r,
            dsize=dsize,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=ignore_index)
    return crop_img, crop_seg

def preprocess_img_gt(img, gt, gt_contour, train_flag):
    img[img>255]=255
    # img = np.array(img, dtype=np.float)
    img = np.array(img, dtype=np.uint8)
    gt = np.array(gt, dtype=np.uint8)
    gt_contour = np.array(gt_contour, dtype=np.uint8)
    if train_flag:
        # img, gt=random_flip(img, gt)
        # img, gt=rand_resize(img, gt)
        # img, gt=random_rotation(crop_img=img, crop_seg=gt, rich_crop_max_rotation=ROTATION, mean_value=93)
        if CROP_FLAG:
            img, gt, gt_contour=rand_crop(img, gt, gt_contour)

    img = np.repeat(img[:,:,None],3,axis=2)
    img = all_transform(img)
    # img=img.repeat([3,1,1])

    gt = torch.Tensor(gt)
    gt = torch.unsqueeze(gt, 0)
    gt_contour = torch.Tensor(gt_contour)
    gt_contour = torch.unsqueeze(gt_contour, 0)
    return img, gt, gt_contour

def make_dataset(root_dir, dataset_dir, val_i):
    train_list, val_list = get_list(dataset_dir, val_i)

    train_dataset = BasicDataset(root_dir, train_list, True)
    val_dataset = BasicDataset(root_dir, val_list, False)

    return train_dataset, val_dataset, train_list, val_list

class BasicDataset(Dataset):
    def __init__(self, root_dir, imgs_list, train_flag):
        self.root_dir = root_dir
        self.imgs_list = imgs_list
        self.train_flag = train_flag

        logging.info(f'Creating dataset with {len(self.imgs_list)} examples')

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, i):
        img = cv2.imread(os.path.join(self.root_dir, self.imgs_list[i][0]), -1)
        gt = cv2.imread(os.path.join(self.root_dir, self.imgs_list[i][1]), -1)
        gt_contour = cv2.imread(os.path.join(self.root_dir, self.imgs_list[i][2]), -1)

        img, gt, gt_contour = preprocess_img_gt(img, gt, gt_contour, self.train_flag)

        return {'image': img, 'mask': gt, 'mask_contour': gt_contour}


############# crop and tile on test img when train img is randomly cropped #########################
def crop_pos(input_size, output_size):
    num_pos = input_size // output_size + 1
    overlap = float(output_size*num_pos-input_size)/float(num_pos-1)
    all_pos = []
    for i in range(num_pos):
        start = i * (output_size-overlap)
        end = start + output_size
        all_pos.append((int(start), int(end)))
    return all_pos

def crop_tile(model, test_img, crop_size=CROP_SIZE):
    # crop test_img according to output_size and tile the result
    # test_img [b, c, h, w]
    if isinstance(crop_size, int):
        crop_size=[CROP_SIZE, CROP_SIZE]
    
    input_h=test_img.shape[2]
    input_w=test_img.shape[3]
    crop_h=crop_size[0]
    crop_w=crop_size[1]
    h_pos=crop_pos(input_h, crop_h)
    w_pos=crop_pos(input_w, crop_w)

    output_prob = []
    for index_i in range(test_img.shape[0]):
        imgi=test_img[index_i]
        img_patchs=[]
        img_count=torch.zeros([input_h, input_w]).cuda()
        for h_posi in h_pos:
            for w_posi in w_pos:
                img_patchs.append(imgi[:, h_posi[0]:h_posi[1], w_posi[0]:w_posi[1]])
                img_count[h_posi[0]:h_posi[1], w_posi[0]:w_posi[1]] += 1
        
        img_patchs = torch.stack(img_patchs, dim=0) #[n, c, crop_h, crop_w]
        prob_patchs = model(img_patchs) #[n, 2, crop_h, crop_w]
        
        patch_i=0
        prob_sum=torch.zeros([prob_patchs.shape[1], input_h, input_w]).cuda()
        for h_posi in h_pos:
            for w_posi in w_pos:
                prob_sum[:,h_posi[0]:h_posi[1], w_posi[0]:w_posi[1]] += prob_patchs[patch_i]
                patch_i+=1

        output_prob.append(prob_sum/img_count)
    
    output = torch.stack(output_prob, dim=0) #[bs, 2, input_h, input_w]
    # return torch.unsqueeze(output, dim=1)
    return output



