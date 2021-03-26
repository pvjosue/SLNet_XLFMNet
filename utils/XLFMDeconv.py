import torch
import nrrd
import sys
from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast,GradScaler
import torchvision as tv
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

from utils.misc_utils import *


def XLFMDeconv(OTF, img, nIt, ObjSize=[512,512], PSFShape=[2160,2160], ROIsize=[512,512],\
                 errorMetric=F.mse_loss, nSplitFourier=1, update_median_limit_multiplier=10, max_allowed=4500, device='cuda:0', all_in_device=False):
    
    reconType = OTF.type()
    nDepths = OTF.shape[1]

    if img.sum()==0:
        volOut = torch.zeros(img.shape[0],nDepths, ObjSize[0], ObjSize[1]).type(reconType)
        proj = volume_2_projections(volOut.permute(0,2,3,1).unsqueeze(1)).cpu()
        return volOut,proj,img,[]

    paddedOTFShape = torch.tensor([ROIsize[i] + PSFShape[i] - 1 for i in range(2)])
    PSFShape = torch.tensor(PSFShape)
    # device = OTF.device

    # Compute transposed OTF
    OTFt = OTF.clone()
    OTFt = torch.real(OTFt) - 1j * torch.imag(OTFt)

    padSize = 2*[(OTF.shape[2] - ObjSize[0])//2] + 2*[(OTF.shape[2] - ObjSize[1])//2]
    padSizeImg = 2*[(OTF.shape[2] - img.shape[2])//2] + 2*[(OTF.shape[2] - img.shape[3])//2]

    # Pad input
    ImgExp = F.pad(img, padSizeImg).to(device)
    with torch.no_grad():
        # Initialize reconstructed volume
        ObjRecon = torch.ones(1,nDepths,ObjSize[0],ObjSize[1])#.type(dtype=reconType)
        
        ImgEst = 0* ImgExp.clone()

        if all_in_device:
            ObjRecon = ObjRecon.to(device)
            OTF = OTF.to(device)
            OTFt = OTFt.to(device)
            ImgEst = ImgEst.to(device)
            ImgExp = ImgExp.to(device)

        losses = []
        end = "\r"
        # plt.ion()
        # plt.figure()
        for ii in range(nIt):
            
            # Compute current image estimate (forward projection)
            ImgEst *= 0.0
            ObjTemp = F.pad(ObjRecon, padSize)
            for jj in range(0,nDepths, nSplitFourier):
                curr_depths = list(range(jj,jj+nSplitFourier))
                planeOTF = OTF[:,curr_depths,...].unsqueeze(1).to(device)
                planeObjFFT = torch.fft.rfft2(ObjTemp[:,curr_depths,...].unsqueeze(1)).to(device)
                ImgEst += F.relu(batch_fftshift2d_real(torch.fft.irfft2(planeObjFFT * planeOTF))).sum(2)
            
            # Compute error in forward image
            ImgEst[ImgEst<1e-6] = 0
            Tmp = ImgExp / (ImgEst+1e-8)    
            if Tmp[Tmp!=0].numel()>0:
                Tmp.clamp_(0.0,Tmp[Tmp!=0].median()*update_median_limit_multiplier)
            Ratio = Tmp.to(device)
            # Propagate error back to volume space and update volume
            for jj in range(0,nDepths, nSplitFourier):
                curr_depths = list(range(jj,jj+nSplitFourier))
                planeObj = ObjTemp[:,curr_depths,...].unsqueeze(1).to(device)
                planeOTF = OTFt[:,curr_depths,...].to(device)
                ObjRecon[:,curr_depths,...] = F.pad(planeObj * batch_fftshift2d_real(torch.fft.irfft2(torch.fft.rfft2(Ratio) * planeOTF)),[-p for p in padSize]).type(ObjRecon.type())
            
            proj = volume_2_projections(ObjRecon.permute(0,2,3,1).unsqueeze(1)).cpu()

            curr_error = errorMetric(ImgExp,ImgEst).item()
            losses.append(curr_error)
            if ii==nIt-1:
                end = "\n"
            print('Deconv it: ' + str(ii+1) + ' / ' + str(nIt) + '\t currErr: ' + str(curr_error), end=end)
            ## Uncomment to show some images
            # plt.subplot(1,3,1)
            # plt.imshow(ImgExp[0,0,...].cpu().numpy())
            # plt.subplot(1,3,2)
            # plt.imshow(ImgEst[0,0,...].cpu().numpy())
            # plt.subplot(1,3,3)
            # plt.imshow(proj[0,0,...].cpu().numpy())
            # plt.pause(0.1)
            # plt.show()
            if max_allowed is not None:
                if ObjRecon.float().max()>=max_allowed:
                    break
    return ObjRecon,proj,ImgEst,losses

