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
train_batch_size=5
val_batch_size=10


train_ds=MyDataset("/Users/vincentc/Desktop/MnM2/train_sa","/Users/vincentc/Desktop/MnM2/train_sa_mask",train_mode=True)
val_ds=MyDataset("/Users/vincentc/Desktop/MnM2/val_sa","/Users/vincentc/Desktop/MnM2/val_sa_mask",train_mode=False)
test_ds=MyDataset("/Users/vincentc/Desktop/MnM2/test_sa","/Users/vincentc/Desktop/MnM2/test_sa_mask",train_mode=False)
#train_ds=MyDataset("./MnM2/MnM2_3d_data/train_sa","./MnM2/MnM2_3d_data/train_sa_mask")
#val_ds=MyDataset("./MnM2/MnM2_3d_data/val_sa","./MnM2/MnM2_3d_data/val_sa_mask")
#test_ds=MyDataset("./MnM2/MnM2_3d_data/test_sa","./MnM2/MnM2_3d_data/test_sa_mask")

train_dl=DataLoader(train_ds,train_batch_size,shuffle=True,pin_memory=True)
val_dl=DataLoader(val_ds,val_batch_size,pin_memory=True)
test_dl=DataLoader(test_ds,val_batch_size,pin_memory=True)
#for img,msk in train_ds:
    #print(img.shape) #1,12,224,224
    #print(msk.shape) #12,224,224
    #msk=msk.unsqueeze(dim=1)
    #img=img.permute(1,2,3,0) #12,224,224,1
    #msk=msk.permute(0,2,3,1) #12,224,224,1
    #print(img.shape)
    #print(msk.shape)
    #for i in range(7):
        #plt.imshow(img[i])
        #plt.show()
        #plt.imshow(msk[i])
        #plt.show()


 
#basic unet
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Down_noz(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d((1,2,2)),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

#x=torch.randn(10,1,12,224,224)
#net=Down(1,4)
#out=net(x)
#print(out.shape)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()

        # if trilinear, use the normal convolutions to reduce the number of channels
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Up_noz(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()

        # if trilinear, use the normal convolutions to reduce the number of channels
        if trilinear:
            self.up = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=(1,2,2), stride=(1,2,2))
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, trilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = trilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down_noz(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if trilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, trilinear)
        self.up2 = Up(512, 256 // factor, trilinear)
        self.up3 = Up(256, 128 // factor, trilinear)
        self.up4 = Up_noz(128, 64, trilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        print('x1',x1.shape)
        x2 = self.down1(x1)
        print('x2',x2.shape)
        x3 = self.down2(x2)
        print('x3',x3.shape)
        x4 = self.down3(x3)
        print('x4',x4.shape)
        x5 = self.down4(x4)
        print('x5',x5.shape)
        x = self.up1(x5, x4)
        print('up1',x.shape)
        x = self.up2(x4, x3)
        print('up2',x.shape)
        x = self.up3(x, x2)
        print('up3',x.shape)
        x = self.up4(x, x1)
        print('up4',x.shape)
        logits = self.outc(x)
        print('output',logits.shape)
        return logits
model=UNet(1,4)

#GPU
def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
def to_device(data,device):
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking=True)

class DeviceDataLoader():
    def __init__(self,dl,device):
        self.dl=dl
        self.device=device
    def __iter__(self):
        for b in self.dl:
            yield to_device(b,self.device)
    def __len__(self):
        return len(self.dl)
device=get_default_device()
model.to(device)
print(next(model.parameters()).is_cuda)

#evaluation
def mIoU(pred_mask, mask, smooth=1e-10, n_classes=23):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(F.softmax(pred_mask, dim=1), dim=1) #shape=(20,244,244)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0: #no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)


def dice_coefficient(pred_mask, mask,smooth=1e-10, n_classes=3):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(F.softmax(pred_mask, dim=1), dim=1) #shape=(20,244,244)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        dice_per_class = []
        for clas in range(0, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas
            #print('class',true_class)
            #print('label',true_label)   
            intersect=torch.logical_and(true_class,true_label).sum().float().item()
            sum=true_class.sum().item()+true_label.sum().item()
            dice_co=2*intersect/sum
            dice_per_class.append(dice_co)
    return np.nanmean(dice_per_class)

def train(epochs,max_lr,model,train_loader,val_loader,weight_decay=0,grad_clip=None,opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    optimizer=opt_func(model.parameters(),max_lr,weight_decay=weight_decay)
    sched=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr,epochs=epochs,steps_per_epoch=len(train_loader))
    history=[]
    max_c=0
    for epoch in range(epochs):
        model.train()
        train_losses=[]
        val_losses=[]
        val_acc=[]

        for images,masks in train_loader:
            #masks = masks.type(torch.LongTensor)
            images=images.to(device)
            masks=masks.to(device)
            out=model(images)
            loss=F.cross_entropy(out,masks)
            loss.backward()
            train_losses.append(loss.item())
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(),grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            sched.step()
            print("train",loss.item())

        with torch.no_grad():
            for images,masks in val_loader:
                #masks=masks.type(torch.LongTensor) #1,8,224,224
                images=images.to(device)
                masks=masks.to(device)
                out=model(images) #1,10,8,224,224
                loss=F.cross_entropy(out,masks)
                print("val",loss.item())
                val_losses.append(loss.item())
                count=0
                for i in range(val_batch_size):
                    dice_score=dice_coefficient(out, masks,smooth=1e-10, n_classes=10)
                    count+=dice_score
                print(count/val_batch_size)
                val_acc.append(count/val_batch_size)


        result=dict()
        result['epoch']=epoch+1
        result['train_loss']=np.mean(train_losses).item()
        result['val_loss']=np.mean(val_losses).item()
        result['val_acc']=np.mean(val_acc).item()

        print('epoch',epoch)
        print('train_loss',result['train_loss'])
        print('val_loss',result['val_loss'])
        print('val_acc',result['val_acc'])
        history.append(result)
    return history

x=train(1,0.01,model,train_dl,val_dl,weight_decay=1e-4, grad_clip=0.1,opt_func=torch.optim.Adam)
