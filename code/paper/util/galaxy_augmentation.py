import json
import numpy as np
import random
from torchvision import transforms
import torch
import cv2
import torchvision.transforms.functional as TF
import scipy.ndimage as ndimage
from PIL import Image

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage


MAX_HW = 384
IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]


# load data from our data folder
data_path = '../../data/GalaxiesV2/'
im_dir = data_path + 'images'

anno_file = data_path + 'annotation.json'
data_split_file = data_path + 'train_test_val.json'

with open(anno_file) as f:
    annotations = json.load(f)

with open(data_split_file) as f:
    data_split = json.load(f)

class resizeTrainImage(object):
    """
    This is our way to adapt CounTR's augmentation pipeline to fit our code. 
    Here we keep most of their steps except mosaicing and blending. 
    
    Resize the image so that:
        1. Image is equal to 384 * 384
        2. The new height and new width are divisible by 16
        3. The aspect ratio is possibly preserved
    Density map is cropped to have the same size(and position) with the cropped image
    Exemplar boxes may be outside the cropped area.
    Augmentation including Gaussian noise, Color jitter, Gaussian blur, Random affine, Random horizontal flip and Random Crop.
    """
    
    def __init__(self, MAX_HW=384):
        self.max_hw = MAX_HW

    def __call__(self, sample):
        image, lines_boxes, density, dots, im_id, m_flag = sample['image'], sample['lines_boxes'],sample['gt_density'], sample['dots'], sample['id'], sample['m_flag']
        
        W, H = image.size

        new_H = 16*int(H/16)
        new_W = 16*int(W/16)
        scale_factor = float(new_W)/ W
        resized_image = transforms.Resize((new_H, new_W))(image)
        resized_density = cv2.resize(density, (new_W, new_H))   
        
        # Augmentation probability
        aug_p = random.random()
        aug_flag = 0
        if aug_p < 0.4: # 0.4
            aug_flag = 1
            if aug_p < 0.25: # 0.25
                aug_flag = 0

        # Gaussian noise
        resized_image = TTensor(resized_image)
        if aug_flag == 1:
            noise = np.random.normal(0, 0.1, resized_image.size())
            noise = torch.from_numpy(noise)
            re_image = resized_image + noise
            re_image = torch.clamp(re_image, 0, 1)

        # Color jitter and Gaussian blur
        if aug_flag == 1:
            re_image = Augmentation(re_image)

        # Random affine
        if aug_flag == 1:
            re1_image = re_image.transpose(0,1).transpose(1,2).numpy()
            keypoints = []
            for i in range(dots.shape[0]):
                keypoints.append(Keypoint(x=min(new_W-1,int(dots[i][0]*scale_factor)), y=min(new_H-1,int(dots[i][1]))))
            kps = KeypointsOnImage(keypoints, re1_image.shape)

            seq = iaa.Sequential([
                iaa.Affine(
                    rotate=(-15,15),
                    scale=(0.8, 1.2),
                    shear=(-10,10),
                    translate_percent={"x": (-0.2,0.2), "y": (-0.2,0.2)}
                )
            ])
            re1_image, kps_aug = seq(image=re1_image, keypoints=kps)

            # Produce dot annotation map
            resized_density = np.zeros((resized_density.shape[0], resized_density.shape[1]),dtype='float32')
            for i in range(len(kps.keypoints)):
                if(int(kps_aug.keypoints[i].y)<= new_H-1 and int(kps_aug.keypoints[i].x)<=new_W-1) and not kps_aug.keypoints[i].is_out_of_image(re1_image):
                    resized_density[int(kps_aug.keypoints[i].y)][int(kps_aug.keypoints[i].x)]=1
            resized_density = torch.from_numpy(resized_density)

            re_image = TTensor(re1_image)

        # Random horizontal flip
        if aug_flag == 1:
            flip_p = random.random()
            if flip_p > 0.5:
                re_image = TF.hflip(re_image)
                resized_density = TF.hflip(resized_density)
        
        # Random 384*384 crop in a new_W*384 image and 384*new_W density map
        if aug_flag == 0:
            re_image = resized_image
            resized_density = np.zeros((resized_density.shape[0], resized_density.shape[1]),dtype='float32')
            for i in range(dots.shape[0]):
                resized_density[min(new_H-1,int(dots[i][1]))][min(new_W-1,int(dots[i][0]*scale_factor))]=1
            resized_density = torch.from_numpy(resized_density)

        start = random.randint(0, new_W-1-383)
        reresized_image = TF.crop(re_image, 0, start, 384, 384)
        reresized_density = resized_density[:, start:start+384]
       
        # Gaussian distribution density map
        reresized_density = ndimage.gaussian_filter(reresized_density.numpy(), sigma=(1, 1), order=0)

        # Density map scale up
        reresized_density = reresized_density * 60
        reresized_density = torch.from_numpy(reresized_density)
            
        # Crop bboxes and resize as 64x64
        boxes = list()
        cnt = 0
        for box in lines_boxes:
            cnt+=1
            if cnt>3:
                break
            box2 = [int(k) for k in box]
            y1, x1, y2, x2 = box2[0], int(box2[1]*scale_factor), box2[2], int(box2[3]*scale_factor)
            bbox = resized_image[:,y1:y2+1,x1:x2+1]
            bbox = transforms.Resize((64, 64))(bbox)
            boxes.append(bbox.numpy())
        boxes = np.array(boxes)
        boxes = torch.Tensor(boxes)
        
        # boxes shape [3,3,64,64], image shape [3,384,384], density shape[384,384]       
        sample = {'image':reresized_image,'boxes':boxes,'gt_density':reresized_density, 'm_flag': m_flag}

        return sample

TTensor = transforms.Compose([   
        transforms.ToTensor(),
        ])

Augmentation = transforms.Compose([   
        transforms.ColorJitter(brightness=0.25, contrast=0.15, saturation=0.15, hue=0.15),
        transforms.GaussianBlur(kernel_size=(7,9))
        ])

Normalize = transforms.Compose([   
        transforms.ToTensor(),
        transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD)
        ])

TransformTrain = transforms.Compose([resizeTrainImage(MAX_HW)])