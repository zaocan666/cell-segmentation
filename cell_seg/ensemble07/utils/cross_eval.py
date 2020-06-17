import os
import random

def cross_split(dataset_dir):
    gt_dir = 'train_GT/SEG'
    assert(os.path.isdir(os.path.join(dataset_dir,gt_dir)))
    img_dir = 'train'
    assert(os.path.isdir(os.path.join(dataset_dir,img_dir)))

    image_path = os.path.join(dataset_dir,img_dir)
    gt_path = os.path.join(dataset_dir,gt_dir)
    images = sorted([os.path.join(img_dir, img) for img in os.listdir(image_path) if img.find('.tif') != -1])
    gts = sorted([os.path.join(gt_dir, gt) for gt in os.listdir(gt_path) if gt.find('.tif') != -1])

    imgs_gts = [[images[i], gts[i]] for i in range(len(images))]
    random.shuffle(imgs_gts)

    vals=[]
    for i in range(len(imgs_gts)):
    	vals.append(i//round(len(imgs_gts)/5.0))

    with open('data_list.txt','w') as f:
    	for i in range(len(imgs_gts)):
    		f.write(imgs_gts[i][0]+' '+imgs_gts[i][1]+' '+str(vals[i])+'\n')

if __name__=='__main__':
	cross_split(r'/home/kaixi_whli/segment/supplementary_modify/dataset1')