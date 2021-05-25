import torch
import sys
from torch.utils import data
from torch.utils.data.sampler import SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast,GradScaler
import torchvision as tv
import torch.nn as nn
import matplotlib.pyplot as plt
import subprocess
import os
import numpy as np
from datetime import datetime
import argparse
import zipfile
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

from nets.XLFMNet import XLFMNet
import utils.pytorch_shot_noise as pytorch_shot_noise
from utils.XLFMDataset import XLFMDatasetFull
from utils.misc_utils import *
from tifffile import imsave

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', nargs='?', default= '')
parser.add_argument('--lenslet_file', nargs='?', default= "lenslet_centers_python.txt")
parser.add_argument('--files_to_store', nargs='+', default=['mainTrainXLFMNet.py','mainTrainSLNet.py','mainCreateDataset.py','utils/XLFMDataset.py','utils/misc_utils.py','nets/extra_nets.py','nets/XLFMNet.py','nets/SLNet.py'])
parser.add_argument('--prefix', nargs='?', default= "fishy")
# parser.add_argument('--checkpoint', nargs='?', default= "/space/vizcainj/shared/XLFMNet/runs/camera_ready_github_Apr/2021_05_03__18:56:00__b'3b27809'_commit__3D_deconv_Final/model_")#runs_dir + 'paper/exp3_SDLFM_3Frame/individual_training/both/2020_11_24__09:24:500_gpu__Aug_unetD2_wf5___1e-4_lessNoise_realImgs_Shuffle_realPower_imgLoss0_MaxPool_DO25_batch8_50imgs1036job/model_')
parser.add_argument('--checkpoint', nargs='?', default= "")

parser.add_argument('--images_to_use', nargs='+', type=int, default=list(range(0,121,1)))
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--use_random_shifts', nargs='+', type=int, default=0, help='Randomize the temporal shifts to use? 0 or 1')
parser.add_argument('--dark_current', type=float, default=106, help='Dark current value of camera.')

parser.add_argument('--output_path', nargs='?', default='out')
parser.add_argument('--main_gpu', nargs='+', type=int, default=[1])

parser.add_argument('--writeToVideo', type=int, default=1)                                
parser.add_argument('--writeVolsToStack', type=int, default=1)
parser.add_argument('--write_to_tb', type=int, default=0)
parser.add_argument('--fps', type=int, default=25.0)
parser.add_argument('--video_multiplyer', type=int, default=2)

debug = False
n_threads = 0
args = parser.parse_args()
if len(args.main_gpu)>0:
    device = "cuda:" + str(args.main_gpu[0])
else:
    device = "cuda"
    args.main_gpu = [1]

if n_threads!=0:
    torch.set_num_threads(n_threads)
torch.manual_seed(261290)


# Load previous checkpoints
if len(args.checkpoint)>0:
    checkpoint_XLFMNet = torch.load(args.checkpoint, map_location=device)
    args_deconv = checkpoint_XLFMNet['args']
    args_SLNet = checkpoint_XLFMNet['args_SLNet']

# Get commit number 
label = subprocess.check_output(["git", "describe", "--always"]).strip()
save_folder = args.output_path + datetime.now().strftime('%Y_%m_%d__%H:%M:%S') + '__' + str(label) + '_commit__' + args.prefix

# Get size of the volume
subimage_shape = args_SLNet.subimage_shape


# Create dataloaders
dataset = XLFMDatasetFull(args.data_folder, args.lenslet_file, subimage_shape, img_shape=[2160,2160],  
            images_to_use=args.images_to_use, divisor=1, isTiff=True, n_frames_net=args_SLNet.n_frames, lenslets_offset=0,
            load_all=True, load_vols=False, load_sparse=False, temporal_shifts=args_SLNet.temporal_shifts, use_random_shifts=args_SLNet.use_random_shifts, eval_video=True)

dataset_size = len(dataset)
test_indices = list(range(dataset_size))
train_sampler = SequentialSampler(dataset)
test_loader = data.DataLoader(dataset, batch_size=args.batch_size, 
                                           sampler=train_sampler, num_workers=0, shuffle=False)

# Get normalization values 
max_images,max_images_sparse,max_volumes = dataset.get_max() 
stats = checkpoint_XLFMNet['statistics']#dataset.get_statistics()
n_lenslets = dataset.len_lenslets()


# Create net
net = XLFMNet(n_lenslets, args_deconv.output_shape, n_temporal_frames=dataset.n_frames, dataset=dataset, 
            use_bias=args_deconv.use_bias, unet_settings=args_deconv.unet_settings).to(device)

# timers
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start_epoch = 0
if len(args.checkpoint)>0:
    net.load_state_dict(checkpoint_XLFMNet['model_state_dict'], strict=False)


# Create summary writer to log stuff
if debug is False:
    writer = SummaryWriter(log_dir=save_folder)
    writer.add_text('arguments',str(vars(args)),0)
    writer.flush()

    # Store files
    zf = zipfile.ZipFile(save_folder + "/files.zip", "w")
    for ff in args.files_to_store:
        zf.write(ff)
    zf.close()

    if args.writeVolsToStack>0:
        os.makedirs(save_folder + '/XLFM_stack_S/')

# Update noramlization stats for SLNet inside network
net.stats = stats
net = net.eval()
# Store times
mean_time = 0

if len(args.main_gpu)>1:
    net = nn.DataParallel(net, args.main_gpu, args.main_gpu[0])


if args.writeToVideo>0:
    video_logs = {'input_frames'    : torch.zeros([1, dataset_size, 1, dataset.img_shape[0], dataset.img_shape[1]]),
                'sparse_reconstruction'     : torch.zeros([1, dataset_size, 1, dataset.img_shape[0], dataset.img_shape[1]]),
                'dense_reconstruction'     : torch.zeros([1, dataset_size, 1, dataset.img_shape[0], dataset.img_shape[1]]),
                'MIP_sparse_3D_reconstruction': torch.zeros([1, dataset_size, 1, args_deconv.output_shape[0] + args_deconv.output_shape[2], args_deconv.output_shape[1] + args_deconv.output_shape[2]])
    }

with torch.no_grad():
    for ix,(curr_img_stack, local_volumes) in enumerate(test_loader):

        # curr_img_stack returns both the dense and the sparse images, here we only need the sparse.
        if len(curr_img_stack.shape)>=5:
        # assert len(curr_img_stack.shape)>=5, "If sparse is used curr_img_stack should contain both images, dense and sparse stacked in the last dim."
            curr_img_sparse = curr_img_stack[...,-1].clone().to(device) 
            curr_img_stack = curr_img_stack[...,0].clone().to(device) 
        else:
            curr_img_sparse = curr_img_stack.clone().to(device) 
        
        curr_img_stack = curr_img_stack.half().to(device)
        local_volumes = local_volumes.half().to(device)

        curr_img_stack -= args.dark_current
        curr_img_stack = F.relu(curr_img_stack).detach()

        curr_img_stack, _ = normalize_type(curr_img_stack, local_volumes, stats['norm_type_img'], stats['mean_imgs'], stats['std_images'], stats['mean_vols'], stats['std_vols'], stats['max_images'], stats['max_vols'])
        

        with autocast():
            start.record()
    
            # Run batch of predicted images in discriminator
            prediction,sparse_prediction = net(curr_img_stack)

            if not all([prediction.shape[i] == subimage_shape[i-2] for i in range(2,4)]):
                diffY = (subimage_shape[0] - prediction.size()[2])
                diffX = (subimage_shape[1] - prediction.size()[3])

                prediction = F.pad(prediction, (diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2))

            # Record training time
            end.record()
            torch.cuda.synchronize()
            end_time = start.elapsed_time(end)
            mean_time += end_time
            
            # Use only first sparse image, corresponding to the selected volume.
            if net_get_params(net).n_frames>1:
                curr_img_sparse = curr_img_sparse[:,0,...].unsqueeze(1)

            if args.writeToVideo>0:
                video_logs['input_frames'][0,ix,...] = curr_img_stack[0,0,...]
                video_logs['sparse_reconstruction'][0,ix,...] = curr_img_sparse[0,0,...]
                video_logs['dense_reconstruction'][0,ix,...] = torch.nn.functional.relu(curr_img_sparse[0,0,...]-curr_img_sparse[0,0,...])
                video_logs['MIP_sparse_3D_reconstruction'][0,ix,...] = volume_2_projections(prediction.permute(0,2,3,1).unsqueeze(1))[0,0,...]
            if args.writeVolsToStack>0:
                imsave(save_folder + '/XLFM_stack_S/XLFM_stack_'+ "%03d" % ix + '.tif', prediction[0,...].cpu().numpy())
            # plt.clf()
            # plt.subplot(1,3,1)
            # plt.imshow(curr_img_stack[0,0,...].float().cpu().detach().numpy())
            # plt.subplot(1,3,2)
            # plt.imshow(curr_img_sparse[0,0,...].float().cpu().detach().numpy())
            # plt.subplot(1,3,3)
            # plt.imshow(volume_2_projections(prediction.permute(0,2,3,1).unsqueeze(1))[0,0,...].float().detach().cpu().numpy())
            # plt.title('Time: '+str(int(end_time)) + ' ms')
            # plt.show()
            # Extract lenslet images
            curr_img_sparse = dataset.extract_views(curr_img_sparse, dataset.lenslet_coords, dataset.subimage_shape)[:,0,...]


        if args.write_to_tb:
            if local_volumes.shape == prediction.shape:
                writer.add_image('max_GT__eval', tv.utils.make_grid(volume_2_projections(local_volumes.permute(0,2,3,1).unsqueeze(1))[0,...], normalize=True, scale_each=True), ix)
                writer.add_image('sum_GT__eval', tv.utils.make_grid(volume_2_projections(local_volumes.permute(0,2,3,1).unsqueeze(1), proj_type=torch.sum)[0,...], normalize=True, scale_each=True), ix)
            

            writer.add_image('max_prediction__eval', tv.utils.make_grid(volume_2_projections(prediction.permute(0,2,3,1).unsqueeze(1))[0,...], normalize=True, scale_each=True), ix)
            writer.add_image('sum_prediction__eval', tv.utils.make_grid(volume_2_projections(prediction.permute(0,2,3,1).unsqueeze(1), proj_type=torch.sum)[0,...], normalize=True, scale_each=True), ix)
            
            # input_noisy_grid = tv.utils.make_grid(curr_img_stack[0,0,...].float().unsqueeze(0).cpu().data.detach(), normalize=True, scale_each=False)

            sparse_prediction = sparse_prediction- sparse_prediction.min()
            sparse_prediction /= sparse_prediction.max()
            input_intermediate_sparse_grid = tv.utils.make_grid(sparse_prediction[0,10,...].float().unsqueeze(0).cpu().data.detach(), normalize=True, scale_each=False)
            input_GT_sparse_grid = tv.utils.make_grid(curr_img_sparse[0,10,...].float().unsqueeze(0).cpu().data.detach(), normalize=True, scale_each=False)

            # volGTHist,volPredHist,inputHist = compute_histograms(local_volumes[0,...].float(), prediction[0,...].float(), curr_img_stack[0,...].float())
            
            # writer.add_image('input_noisy_'+curr_train_stage, input_noisy_grid, ix)
            writer.add_image('image_intermediate_sparse_eval', input_intermediate_sparse_grid, ix)
            writer.add_image('image_intermediate_sparse_GT_eval', input_GT_sparse_grid, ix)
            # writer.add_scalar('Loss/_eval', ix)
            # writer.add_scalar('psnr/_eval', mean_psnr, ix)
            writer.add_scalar('times/_eval', mean_time, ix)
            

    if args.writeToVideo==1:
        from cv2 import VideoWriter, VideoWriter_fourcc
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        for vid_path,video_tensor in video_logs.items():
            video_tensor.clamp_(min=0, max=video_tensor.max().item()/args.video_multiplyer)
            video_tensor /= video_tensor.max()
            
            curr_writer = VideoWriter(save_folder + '/' + vid_path + '.avi', fourcc, args.fps, (video_tensor.shape[4], video_tensor.shape[3]), 0)
            for nFrame in range(video_tensor.shape[1]):
                gray = (255*video_tensor[0,nFrame,0,...]).numpy().astype(np.uint8)#cv2.normalize(video_input[0,nFrame,0,...].numpy(), None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                curr_writer.write(gray)
            curr_writer.release()

            
        

