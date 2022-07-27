import numpy as np
import torch
import torch.nn.functional as F

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