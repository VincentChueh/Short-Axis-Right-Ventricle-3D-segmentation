from statistics import mode
import numpy as np
import torch
import nibabel as nib
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
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

        #normalization
        norm=tio.transforms.ZNormalization()
        image=norm(image)
        #mask=norm(mask)

        if self.train_mode ==True:
            #elastic_deform = tio.transforms.RandomElasticDeformation(num_control_points=(4,4,4), locked_borders=1,)
            #image=elastic_deform(image)
            #mask=elastic_deform(mask)

            #flip=tio.transforms.RandomFlip(axes=('Left',))
            #image=flip(image)
            #mask=flip(mask)
            pass
        else:
            pass
        mask=mask.squeeze(dim=0)
        return image,mask