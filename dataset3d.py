from statistics import mode
import numpy as np
import torch
import nibabel as nib
import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler
import torchio as tio


class MyDataset(Dataset):
    def __init__(self,image_dir,mask_dir,transform=None,train_mode=True):
        self.image_dir=image_dir
        self.mask_dir=mask_dir
        self.transform=transform
        self.train_mode=train_mode
        self.masks=os.listdir(mask_dir)

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, index):
        mask_path=os.path.join(self.mask_dir,self.masks[index])
        img_path=os.path.join(self.image_dir,self.masks[index].replace("_gt.nii.gz",".nii.gz"))
        img=nib.load(img_path)
        image=np.array(img.dataobj)
        image=image.astype(np.float32)
        msk=nib.load(mask_path) 
        mask=np.array(msk.dataobj)
        mask=mask.astype(np.float32)
        image=torch.from_numpy(image).permute(2,0,1).unsqueeze(dim=0) #1,12,224,224
        mask=torch.from_numpy(mask).permute(2,0,1).unsqueeze(dim=0).to(dtype=torch.long) #12,224,224

        #reshape z dim
        reshape = tio.transforms.CropOrPad((8, 224, 224))
        image = reshape(image)
        mask = reshape(mask)
        
        if self.train_mode ==True:
            #clamp
            clamp = tio.Clamp(out_min=-300, out_max=300,p=1)
            image=clamp(image)
        
            #resample
            resample = tio.Resample((1,1,1),p=0.5)
            image=resample(image)

            #blur
            blur=tio.transforms.RandomBlur(std=(0,2),p=1)
            image=blur(image)
        
            #noise
            noise=tio.transforms.RandomNoise(mean=0,std=(0,80),p=1)
            image=noise(image)
            
            #normalization
            norm=tio.transforms.ZNormalization()
            image=norm(image)       
       
        else:
            #normalization
            norm=tio.transforms.ZNormalization()
            image=norm(image)


        mask=mask.squeeze(dim=0)
        return image,mask