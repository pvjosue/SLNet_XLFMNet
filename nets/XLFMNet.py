import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast,GradScaler
# from nets.unet_shallow import UNetLF
from nets.extra_nets import UNet as UNetLF
from nets.SLNet import *
from utils.XLFMDataset import XLFMDatasetFull
from utils.misc_utils import *

class XLFMNet(nn.Module):
    def __init__(self, in_views, output_shape, n_temporal_frames=1, dataset=None, use_bias=False, stats={}, unet_settings={'depth':5, 'wf':6, 'drop_out':1.0, 'batch_norm':True}):
        super(XLFMNet, self).__init__()
        self.output_shape = output_shape
        self.n_frames = n_temporal_frames

        self.dataset = dataset
        self.stats = stats
        out_depths = output_shape[2]
        
        # Merge nFrames before normal XLFMNet
        if self.n_frames!= 1:
            # set the frames on the 3D dimension, so a 3D convolution can use all time steps at once
            self.tempConv = SLNet(n_temporal_frames=n_temporal_frames)
        
        # 3D reconstruction net
        self.deconv = nn.Sequential(
                        nn.Conv2d(in_views,out_depths, 3, stride=1, padding=1, bias=use_bias),
                        nn.BatchNorm2d(out_depths),
                        nn.LeakyReLU(),
                        UNetLF(out_depths, out_depths, depth=unet_settings['depth'], wf=unet_settings['wf'], drop_out=unet_settings['drop_out'], use_bias=use_bias))
        

    @autocast()
    def forward(self, input):
        # Fetch normalization stats for SLNet
        stats = self.stats
        intermediate_result = input
        # Compute sparse input with SLNet
        if self.n_frames!= 1 and self.tempConv is not None:
            D = self.tempConv(input)
            sparse_part = F.relu(input.detach() - D)
            # if len(stats)>0:
                # sparse_part, _ = normalize_type(sparse_part, torch.zeros([1]), stats['norm_type'], stats['mean_imgs'], stats['std_images'], 0, 1, stats['max_images'], 10, inverse=True)
                # D, _ = normalize_type(D, torch.zeros([1]), stats['norm_type'], stats['mean_imgs'], stats['std_images'], 0, 1, stats['max_images'], 10, inverse=True)
                # input, _ = normalize_type(input, torch.zeros([1]), stats['norm_type'], stats['mean_imgs'], stats['std_images'], 0, 1, stats['max_images'], 10, inverse=True)
                # sparse_part = F.relu(input-D.detach())
        else:
            sparse_part = input
        intermediate_result = self.dataset.extract_views(sparse_part[:,0,...].unsqueeze(1), self.dataset.lenslet_coords, self.dataset.subimage_shape)[:,0,...]
            
        # Run 3D reconstruction network
        out = self.deconv(intermediate_result)
            
        return out, intermediate_result
