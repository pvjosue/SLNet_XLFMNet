import torch
import nrrd
import sys
from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils import data
from torch.utils.data.sampler import SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
import torchvision as tv
import torch.nn as nn
import matplotlib.pyplot as plt
import subprocess
import glob, os
import numpy as np
from datetime import datetime
import argparse
import zipfile
from shutil import copyfile
from tifffile import imsave

import utils.pytorch_shot_noise as pytorch_shot_noise
from utils.XLFMDataset import XLFMDatasetFull
from utils.misc_utils import *
from utils.XLFMDeconv import *
from nets.SLNet import *

main_folder = "/u/home/vizcainj/code/XLFMNet/"
runs_dir = "/space/vizcainj/shared/XLFMNet/runs/"
data_dir = "/space/vizcainj/shared/datasets/XLFM/"
# Real image 
# filename = "20200903_NLS_GCaMP6s_XLFM_confocal10x/XLFM/all_images"
filename = "20201111_test_fish/fish2_new"
# filename = "20201111_test_fish/fish3_new5outScaleS_Jan19"

check = '/space/vizcainj/shared/XLFMNet/runs/camera_ready_github/2021_05_17__16:55:200_gpu__Fish2/model_300'


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', nargs='?', default= data_dir + "/XLFM_real_fish/" + filename, help='Input images path in format /XLFM_image/XLFM_image_stack.tif and XLFM_image_stack_S.tif in case of a sparse GT stack.')
parser.add_argument('--lenslet_file', nargs='?', default= "lenslet_centers_python.txt")
parser.add_argument('--files_to_store', nargs='+', default=[], help='Relative paths of files to store in a zip when running this script, for backup.')
parser.add_argument('--prefix', nargs='?', default= "fish2_new", help='Prefix string for the output folder.')
parser.add_argument('--checkpoint', nargs='?', default= check, help='File path of checkpoint of SLNet.')
parser.add_argument('--psf_file', nargs='?', default= main_folder + "/data/20200730_XLFM_beads_images/20200730_XLFM_PSF_2.5um/PSF_2.5um_processed.mat", help='.mat matlab file with PSF stack, used for deconvolution.')
# Images related arguments
parser.add_argument('--images_to_use', nargs='+', type=int, default=list(range(0,193,1)), help='Indeces of images to train on.')
parser.add_argument('--n_simulations', type=int, default=90, help='Number of samples to generate.')
# Noise arguments
parser.add_argument('--add_noise', type=int, default=0, help='Apply noise to images? 0 or 1')
parser.add_argument('--signal_power_max', type=float, default=30**2, help='Max signal value to control signal to noise ratio when applyting noise.')
parser.add_argument('--signal_power_min', type=float, default=60**2, help='Min signal value to control signal to noise ratio when applyting noise.')
parser.add_argument('--dark_current', type=float, default=106, help='Dark current value of camera.')
parser.add_argument('--dark_current_sparse', type=float, default=0, help='Dark current value of camera.')

# Sparse decomposition arguments
parser.add_argument('--temporal_shifts', nargs='+', type=int, default=[0,49,99], help='Which frames to use for training and testing.')
parser.add_argument('--SD_iterations', type=int, default=0, help='Number of iterations for Sparse Decomposition, 0 to disable.')
parser.add_argument('--frame_to_grab', type=int, default=0, help='Which frame to show from the sparse decomposition?')
# 3D deconvolution arguments
parser.add_argument('--deconv_iterations', type=int, default=50, help='Number of iterations for 3D deconvolution, for GT volume generation.')
parser.add_argument('--deconv_n_depths', type=int, default=120, help='Number of depths to create in 3D deconvolution.')
parser.add_argument('--n_depths', type=int, default=120, help='Number of depths to create in 3D deconvolution.')
parser.add_argument('--deconv_limit', type=float, default=10000, help='Maximum intensity allowed from doconvolution.')
parser.add_argument('--deconv_depth_split', type=int, default=60, help='Number of depths to simultaneously deconvolve in the gpu.')
parser.add_argument('--deconv_gpu', type=int, default=2, help='GPU to use for deconvolution, -1 to use CPU, this is very memory intensive.')    

parser.add_argument('--output_path', nargs='?', default='/space/vizcainj/shared/datasets/XLFM/camera_ready/')
# parser.add_argument('--output_path', nargs='?', default=runs_dir + '/garbage/')
parser.add_argument('--main_gpu', nargs='+', type=int, default=[5], help='List of GPUs to use: [0,1]')

# Set to zero if debuging
n_threads = 0
args = parser.parse_args()
# Select which devices to use
if len(args.main_gpu)>0:
    device = "cuda:" + str(args.main_gpu[0])
else:
    if args.main_gpu==-1:
        device = "cpu"
    else:
        device = "cuda"
        args.main_gpu = [0]

# Deconvolution can be heavy on the GPU, and sometimes it doesn't fit, so use -1 for CPU
if args.deconv_gpu==-1:
    device_deconv = "cpu"
else:
    device_deconv = "cuda:" + str(args.deconv_gpu)

if n_threads!=0:
    torch.set_num_threads(n_threads)

checkpoint_path = None
if len(args.checkpoint)>0:
    checkpoint = torch.load(args.checkpoint, map_location=device)
    currArgs = args
    argsModel = checkpoint['args']


# Get images
dataset = XLFMDatasetFull(args.data_folder, args.lenslet_file, argsModel.subimage_shape, img_shape=2*[argsModel.img_size],
            images_to_use=args.images_to_use, divisor=1, isTiff=True, n_frames_net=argsModel.n_frames, 
            load_all=True, load_sparse=False, load_vols=False, temporal_shifts=args.temporal_shifts, eval_video=True)

# Get normalization values 
max_images,max_images_sparse,max_volumes = dataset.get_max()
# Normalization from SLNet
mean_imgs,std_images,mean_vols,std_vols = checkpoint['statistics']
mean_imgs = mean_imgs.to(device)
std_images = std_images.to(device)
mean_vols = mean_vols.to(device)
std_vols = std_vols.to(device)

n_images = len(dataset)

# Get volume desired shape
output_shape = argsModel.output_shape + [args.n_depths]
if len(output_shape)==2:
    output_shape = argsModel.output_shape + [args.n_depths]

# Creating data indices for training and validation splits:
dataset_size = len(dataset)

# Create dataloader
train_sampler = SequentialSampler(dataset)

# Create output directory
head, tail = os.path.split(args.checkpoint)
output_dir = head + '/Dataset_' + datetime.now().strftime('%Y_%m_%d__%H:%M:%S') + '_' + str(args.n_depths) + 'nD__' + str(args.n_simulations) + 'nS__' + args.prefix
print('Output directory: ' + output_dir)
# Create directories
# XLFM_image: Raw XLFM image
# XLFM_stack: Deconvolution of raw images (not computed here)
# XLFM_stack_S: Deconvolution of sparse image generated by SD algorithm
# XLFM_stack_SL: Deconvolution of sparse image generated by SLNet
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    os.mkdir(output_dir + '/XLFM_image')
    os.mkdir(output_dir + '/XLFM_stack')
    os.mkdir(output_dir + '/XLFM_stack_S')
    os.mkdir(output_dir + '/XLFM_stack_S_SL')

# Tensorboard logger
writer = SummaryWriter(output_dir)
writer.add_text('arguments',str(vars(args)),0)
writer.add_text('arguments_model',str(vars(checkpoint['args'])),0)
writer.flush()

# Copy files and model to output folder
try:
    copyfile(os.path.dirname(args.checkpoint)+'/files.zip', output_dir+'/files.zip')
    copyfile(args.checkpoint, output_dir + '/' + tail)
    # Extract the files used for training SLNet, a different type of version control
    with zipfile.ZipFile(os.path.dirname(args.checkpoint)+'/files.zip', "r") as zip_ref:
        os.makedirs('tmp_files', exist_ok=True)
        zip_ref.extractall('tmp_files/')
        from tmp_files.nets.SLNet import *
except:
    pass
    

# Create net and load checkpoint
net = SLNet(dataset.n_frames, mu_sum_constraint=argsModel.SL_mu_sum_constraint, alpha_l1=argsModel.SL_alpha_l1).to(device)
net.eval()

if 'module' in list(checkpoint['model_state_dict'])[0]:
    net = nn.DataParallel(net)
    net.load_state_dict(checkpoint['model_state_dict'])
    net = net.module
else:
    net.load_state_dict(checkpoint['model_state_dict'])

# timers
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
end_time_deconv_SL = 0
end_time_deconv_net = 0
# Load measured PSF from matlab file and compute OTF
# The computetion can be splited to fit it into the GPU, spliting n_depths/n_split
psf_shape = 2*[argsModel.img_size]
if currArgs.deconv_iterations > 0:
    n_split = 20
    OTF,psf_shape = load_PSF_OTF(currArgs.psf_file, output_shape, n_depths=args.deconv_n_depths, 
                                n_split=n_split, lenslet_centers_file_out="", compute_transpose=True)



# Create array to gather all images, which contains:
# 0: input
# 1: dense SLNet
# 2: sparse SLNet
# 3: dense SL
# 4: sparse SL
tmp = 3
if args.SD_iterations > 0:
    tmp = 5
all_images = np.zeros((tmp, args.n_simulations*len(args.temporal_shifts)) + tuple(psf_shape), 'float16')

# Compute images
with torch.no_grad():
    mean_time = 0
    mean_time_SL = 0
    min_time = 10000.0
    # Training
    for nSimul in range(args.n_simulations):
        print('Simulating ' + str(nSimul) + ' / ' + str(args.n_simulations) )
        curr_index = nSimul%n_images
        
        # fetch current pair
        curr_img_stack, local_volumes = dataset.__getitem__(curr_index)
        curr_img_stack = curr_img_stack.unsqueeze(0)

        curr_img_stack = curr_img_stack.float()
        # curr_img_stack = curr_img_stack / curr_img_stack.max() * 3000.0
        curr_img_stack = curr_img_stack.half()

        curr_img_stack = curr_img_stack.to(device)

        if len(curr_img_stack.shape)>=5:
        # assert len(curr_img_stack.shape)>=5, "If sparse is used curr_img_stack should contain both images, dense and sparse stacked in the last dim."
            curr_img_sparse = curr_img_stack[...,-1].clone().to(device) 
            curr_img_stack = curr_img_stack[...,0].clone() 
        else:
            curr_img_sparse = curr_img_stack.clone()
        raw_image_stack = curr_img_stack.clone() 

        # Remove dark current from images
        curr_img_stack -= args.dark_current
        curr_img_stack = F.relu(curr_img_stack).detach()
        curr_img_sparse -= args.dark_current_sparse
        curr_img_sparse = F.relu(curr_img_sparse).detach()

        if args.add_noise==1:
            curr_max = curr_img_stack.max()
            # Update new signal power
            signal_power = (args.signal_power_min + (args.signal_power_max-args.signal_power_min) * torch.rand(1)).item()
            curr_img_stack = signal_power/curr_max * curr_img_stack
            # Add noise
            curr_img_stack = pytorch_shot_noise.add_camera_noise(curr_img_stack)
            curr_img_stack = curr_max/signal_power * curr_img_stack.to(device)
        
        # Normalize images with the same settings as the SLNet was trained
        curr_img_stack, local_volumes = normalize_type(curr_img_stack, local_volumes, argsModel.norm_type, mean_imgs, std_images, mean_vols, std_vols, max_images, max_volumes)
        

        with autocast():
            # torch.cuda.synchronize()
            start.record()
            # Run batch of predicted images in discriminator
            dense_part = net(curr_img_stack)
            # Record training time
            end.record()
            torch.cuda.synchronize()
            end_time = start.elapsed_time(end) / curr_img_stack.shape[0]
            mean_time += end_time
            min_time = min(min_time, end_time)

            sparse_part = F.relu(curr_img_stack - dense_part)
            # Renormalize images
            dense_part, _ = normalize_type(dense_part.float(), local_volumes, argsModel.norm_type, mean_imgs, std_images, mean_vols, std_vols, max_images, max_volumes, inverse=True)
            sparse_part, _ = normalize_type(sparse_part.float(), local_volumes, argsModel.norm_type, mean_imgs, std_images, mean_vols, std_vols, max_images, max_volumes, inverse=True)
            curr_img_stack, _ = normalize_type(curr_img_stack.float(), local_volumes, argsModel.norm_type, mean_imgs, std_images, mean_vols, std_vols, max_images, max_volumes, inverse=True)
            sparse_part = F.relu(curr_img_stack-dense_part.detach())

        # Deconvolve the SLNet sparse image and store the 3D stack
        if currArgs.deconv_iterations > 0:
            start.record()
            img_to_deconv_net = sparse_part[:,currArgs.frame_to_grab].unsqueeze(1).float()
            deconv_net,proj_net,forward_net,_ = XLFMDeconv(OTF, img_to_deconv_net, currArgs.deconv_iterations, 
                                                device=device_deconv, all_in_device=0, 
                                                nSplitFourier=args.deconv_depth_split,max_allowed=args.deconv_limit)
            end.record()
            torch.cuda.synchronize()
            end_time_deconv_net = start.elapsed_time(end) / curr_img_stack.shape[0]
            deconv_net = deconv_net[:, currArgs.deconv_n_depths//2-currArgs.n_depths//2 : currArgs.deconv_n_depths//2+currArgs.n_depths//2,...]
            print(end_time_deconv_net,'s  ',str(deconv_net.max()))
            imsave(output_dir + '/XLFM_stack_S/XLFM_stack_'+ "%03d" % nSimul + '.tif', deconv_net.cpu().numpy())

        # Generate GT with SL decomposition and deconvolve it
        if args.SD_iterations > 0:
            dense_part_SL,sparse_part_SL,_ = SLDecomposition(curr_img_stack, maxIter=args.SD_iterations)
            end.record()
            torch.cuda.synchronize()
            end_time_SL = start.elapsed_time(end) / curr_img_stack.shape[0]
            mean_time_SL += end_time_SL

            all_images[3,nSimul,:,:] = dense_part_SL[0,args.frame_to_grab,...].cpu().numpy().astype(np.float16)
            all_images[4,nSimul,:,:] = sparse_part_SL[0,args.frame_to_grab,...].cpu().numpy().astype(np.float16)
            if currArgs.deconv_iterations > 0:
                start.record()
                img_to_deconv_SL = sparse_part_SL[:,currArgs.frame_to_grab].unsqueeze(1).float()
                deconv_SL,proj_SL,forward_SL,_ = XLFMDeconv(OTF, img_to_deconv_SL, currArgs.deconv_iterations, device=device, all_in_device=args.deconv_gpu)
                end.record()
                torch.cuda.synchronize()
                end_time_deconv_SL = start.elapsed_time(end) / curr_img_stack.shape[0]

                deconv_SL = deconv_SL[:, currArgs.deconv_n_depths//2-currArgs.n_depths//2 : currArgs.deconv_n_depths//2+currArgs.n_depths//2,...]

                imsave(output_dir + '/XLFM_stack_S_SL/XLFM_stack_'+ "%03d" % nSimul + '.tif', deconv_SL.cpu().numpy())
            # set to true for plotting
            if False:
                plot_multiplier = 4
                plt.subplot(2,3,1)
                plt.imshow(img_to_deconv_SL[0,0,...].cpu().numpy())
                plt.title('Input SL')
                plt.subplot(2,3,2)
                plt.imshow(forward_SL[0,0,...].cpu().numpy())
                plt.title('forward SL')
                plt.subplot(2,3,3)
                plt.imshow(proj_SL[0,0,...].clamp(0,proj_SL.max()/plot_multiplier).cpu().numpy())
                plt.title('proj SL')

                plt.subplot(2,3,4)
                plt.imshow(img_to_deconv_net[0,0,...].clamp(0,img_to_deconv_net.max()/plot_multiplier).cpu().numpy())
                plt.title('Input SLNet')
                plt.subplot(2,3,5)
                plt.imshow(forward_net[0,0,...].clamp(0,forward_net.max()/plot_multiplier).cpu().numpy())
                plt.title('forward SLNet')
                plt.subplot(2,3,6)
                plt.imshow(proj_net[0,0,...].clamp(0,proj_net.max()/plot_multiplier).cpu().numpy())
                plt.title('proj SLNet')
                plt.show()
        

        # Store current images
        curr_img_ix = nSimul * len(args.temporal_shifts)
        all_images[0,curr_img_ix:curr_img_ix+len(args.temporal_shifts),...] = raw_image_stack.cpu().numpy().astype(np.float16)
        all_images[1,curr_img_ix:curr_img_ix+len(args.temporal_shifts),...] = dense_part.cpu().numpy().astype(np.float16)
        all_images[2,curr_img_ix:curr_img_ix+len(args.temporal_shifts),...] = sparse_part.cpu().numpy().astype(np.float16)

              

        rescale_img = lambda img: F.interpolate( img, [img.shape[-2]//10, img.shape[-1]//10])
        
        # Store images and log them to tensorboard
        img_labels = ['input','dense_SLNet','sparse_SLNet','dense_SL','sparse_SL']
        for nImg in range(all_images.shape[0]):
            img = tv.utils.make_grid(rescale_img(torch.from_numpy(all_images[nImg,args.frame_to_grab+nSimul*len(args.temporal_shifts),...]).float().unsqueeze(0).unsqueeze(0).cpu().detach()), normalize=True, scale_each=False)
            writer.add_image(img_labels[nImg],img, nSimul)
        
        if "proj_net" in locals():
            writer.add_image('proj_SLNet',tv.utils.make_grid(proj_net[0,...].cpu(), normalize=True), nSimul)
        if "proj_SL" in locals():
            writer.add_image('proj_SL',tv.utils.make_grid(proj_SL[0,...].cpu(), normalize=True), nSimul)
        writer.add_scalar('times/Net', end_time, nSimul)
        if args.SD_iterations>0:
            writer.add_scalar('times/SL', end_time_SL, nSimul)

        if args.deconv_iterations>0:
            writer.add_scalar('times/deconv_SL', end_time_deconv_SL, nSimul)
            writer.add_scalar('times/deconv_SLNet', end_time_deconv_net, nSimul)
        
img_labels = [  'XLFM_image_stack.tif',
                'XLFM_image_stack_D_SLNet.tif',
                'XLFM_image_stack_S.tif',
                'XLFM_image_stack_D_SL.tif',
                'XLFM_image_stack_S_SL.tif']

for nImg in range(all_images.shape[0]):
    imsave(output_dir + '/XLFM_image/' + img_labels[nImg], all_images[nImg,...])

# Store computation times in tensorboard
writer.add_scalar('mean/time', mean_time)
writer.add_scalar('mean/time_SL', mean_time_SL)
if args.deconv_iterations>0:
    writer.add_scalar('mean/time_deconv', end_time_deconv_net)
if args.SD_iterations>0:
    writer.add_scalar('mean/time_SL', mean_time_SL)
if args.deconv_iterations>0:
    writer.add_scalar('mean/time_deconv_SL', end_time_deconv_SL)
writer.add_scalar('mean/min_time', min_time)

writer.close()
            
        
