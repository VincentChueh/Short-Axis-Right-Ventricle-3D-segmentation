import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler
from .evaluation import mIoU, dice_coefficient
from .to_gpu import get_default_device


device=get_default_device()
val_batch_size=10

def train(epochs,max_lr,model,train_loader,val_loader,weight_decay=0,grad_clip=None,opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    print('begin')
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

        with torch.no_grad():
            for images,masks in val_loader:
                images=images.to(device)
                masks=masks.to(device)
                out=model(images) #1,10,8,224,224
                loss=F.cross_entropy(out,masks)
                print("val",loss.item())
                val_losses.append(loss.item())
                count=0
                for i in range(val_batch_size):
                    dice_score=dice_coefficient(out, masks,smooth=1e-10, n_classes=3)
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

        #save model
        criteria=result['val_acc']
        if max_c<criteria:
            max_c=criteria
            torch.save(model,'./3d_ce500.pt')
            print('model saved! Criteria=',criteria)

    return history
