import random
import torch

from torchvision.transforms import functional as F
import torchvision
import numpy as np
from PIL import Image

def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data

def _flip_lsp_person_keypoints(kps, width):
    flip_inds = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13]
    #flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]

    #In LSP it seems that either the visibility = 0 means visible or that
    #it is wrongly tagged. Therefore, we will not use it. 

    # Maintain COCO convention that if visibility == 0, then x, y = 0
    #inds = flipped_data[..., 2] == 0

    #If using LSP visibility = 0 (visible) and 1 (non-visible):
    inds = flipped_data[..., 2] == 1

    #flipped_data[inds] = 0
    return flipped_data
    
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_lsp_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target

class RandomRotation(object):
    def __init__(self, rot_left, rot_right,rot_down):
        self.rot_left = rot_left
        self.rot_right = rot_right
        self.rot_down = rot_down

    def __call__(self, image, target):
        rand_value = random.random() #retorna un valor entre [0 i 1)

        if rand_value <= self.rot_left:
            height, width = image.shape[-2:]
            #la imatge està en tensor amb size torch.Size([3, 160, 70]), per tant només rotem el [1,2]. 
            #Com rota cap a l'esquerra, només rotem 1 cop (=90º esquerra)
            image = torch.rot90(image,1,[1,2]) 
            boxes = target["boxes"]
            boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3] = boxes[:,1],width-boxes[:,0],boxes[:,3],width-boxes[:,2]
            target["boxes"] = boxes
            keypoints = target["keypoints"]
            keypoints[...,0], keypoints[...,1] = keypoints[...,1], width - keypoints[...,0]
            target["keypoints"] = keypoints

        elif (self.rot_left < rand_value and rand_value <= (self.rot_left + self.rot_right)):
            height, width = image.shape[-2:]
            #la imatge està en tensor amb size torch.Size([3, 160, 70]), per tant només rotem el [1,2]. 
            #Com rota cap a l'esquerra, rotem 2 cops (=270º esquerra = 90º dreta)
            image = torch.rot90(image,3,[1,2])
            boxes = target["boxes"]
            boxes[:,1],boxes[:,0],boxes[:,3],boxes[:,2] = boxes[:,0],height-boxes[:,1],boxes[:,2],height-boxes[:,3]
            target["boxes"] = boxes
            keypoints = target["keypoints"]
            keypoints[...,1], keypoints[...,0] = keypoints[...,0],height - keypoints[...,1]
            target["keypoints"] = keypoints

        elif ((self.rot_right+self.rot_left) < rand_value and rand_value <= (self.rot_left + self.rot_right + self.rot_down)):
            height, width = image.shape[-2:]
            #la imatge està en tensor amb size torch.Size([3, 160, 70]), per tant només rotem el [1,2]. 
            #Com rota cap a l'esquerra, rotem 2 cops (=270º esquerra = 90º dreta)
            image = torch.rot90(image,2,[1,2])
            boxes = target["boxes"]
            boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3] = width-boxes[:,0],height-boxes[:,1],width-boxes[:,2],height-boxes[:,3]
            target["boxes"] = boxes
            keypoints = target["keypoints"]
            keypoints[...,0], keypoints[...,1] = width-keypoints[...,0],height - keypoints[...,1]
            target["keypoints"] = keypoints

        return image, target
    
class Albumentation(object):
    def __init__(self, transform, p):
        self.p = p
        self.t = transform
    def __call__(self, image, target):
        rand_value = random.random() #retorna un valor entre [0 i 1)

        if rand_value <= self.p:
          
          #Convert Tensor to Numpy (through PIL)
          im = torchvision.transforms.ToPILImage()(image).convert("RGB")  #Tensor to PIL
          image_np = np.array(im)                                         #PIL to numpy

          # Apply transformations
          augmented_image = self.t(image=image_np)

          #Convert Numpy to Tensor (through PIL)
          image = Image.fromarray(augmented_image['image'])               #numpy to PIL
          image = F.to_tensor(image)                                      #PIL to Tensor

        return image, target
    
class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
