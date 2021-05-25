import torch
import sys
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
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

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', nargs='?', default= '')
parser.add_argument('--data_folder_test', nargs='?', default='')
parser.add_argument('--lenslet_file', nargs='?', default= "lenslet_centers_python.txt")
parser.add_argument('--files_to_store', nargs='+', default=['mainTrainXLFMNet.py','mainTrainSLNet.py','mainCreateDataset.py','utils/XLFMDataset.py','utils/misc_utils.py','nets/extra_nets.py','nets/XLFMNet.py','nets/SLNet.py'])
parser.add_argument('--psf_file', nargs='?', default= "PSF_2.5um_processed.mat")
parser.add_argument('--prefix', nargs='?', default= "fishy")
parser.add_argument('--checkpoint', nargs='?', default= "")
parser.add_argument('--checkpoint_XLFMNet', nargs='?', default= "")
parser.add_argument('--checkpoint_SLNet', nargs='?', default="")

parser.add_argument('--images_to_use', nargs='+', type=int, default=list(range(0,50,1)))
parser.add_argument('--images_to_use_test', nargs='+', type=int, default=list(range(0,10,1)))
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--max_epochs', type=int, default=501)
parser.add_argument('--validation_split', type=float, default=0.1)
parser.add_argument('--eval_every', type=int, default=25)
parser.add_argument('--shuffle_dataset', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--use_bias', type=int, default=0)
parser.add_argument('--use_random_shifts', nargs='+', type=int, default=0, help='Randomize the temporal shifts to use? 0 or 1')
# Noise arguments
parser.add_argument('--add_noise', type=int, default=0, help='Apply noise to images? 0 or 1')
parser.add_argument('--signal_power_max', type=float, default=30**2, help='Max signal value to control signal to noise ratio when applyting noise.')
parser.add_argument('--signal_power_min', type=float, default=60**2, help='Min signal value to control signal to noise ratio when applyting noise.')
parser.add_argument('--norm_type', type=float, default=1, help='Normalization type, see the normalize_type function for more info.')
parser.add_argument('--dark_current', type=float, default=106, help='Dark current value of camera.')
parser.add_argument('--dark_current_sparse', type=float, default=0, help='Dark current value of camera.')

parser.add_argument('--use_sparse', type=int, default=1)
parser.add_argument('--use_img_loss', type=float, default=1.0)

parser.add_argument('--unet_depth', type=int, default=2)
parser.add_argument('--unet_wf', type=int, default=7)
parser.add_argument('--unet_drop_out', type=float, default=0)

parser.add_argument('--output_path', nargs='?', default='')
parser.add_argument('--main_gpu', nargs='+', type=int, default=[1])
parser.add_argument('--gpu_repro', nargs='+', type=int, default=[])
parser.add_argument('--n_split', type=int, default=20)

debug = False
n_threads = 0
args = parser.parse_args()
if len(args.main_gpu)>0:
    device = "cuda:" + str(args.main_gpu[0])
    device_repro = "cuda:" + str(args.main_gpu[0]+1)
else:
    device = "cuda"
    device_repro = "cuda"
    args.main_gpu = [1]
    args.gpu_repro = [1]

if len(args.gpu_repro)==0:
    device_repro = "cpu"
else:
    device_repro = "cuda:" + str(args.gpu_repro[0])

if n_threads!=0:
    torch.set_num_threads(n_threads)
torch.manual_seed(261290)


# Load previous checkpoints
if len(args.checkpoint_XLFMNet)>0:
    checkpoint_XLFMNet = torch.load(args.checkpoint_XLFMNet, map_location=device)
    args_deconv = checkpoint_XLFMNet['args']
    args.unet_depth = args_deconv.unet_depth
    args.unet_wf = args_deconv.unet_wf


if len(args.checkpoint_SLNet)>0:
    checkpoint_SL = torch.load(args.checkpoint_SLNet, map_location=device)
    argsSLNet = checkpoint_SL['args']
    args.temporal_shifts = checkpoint_SL['args'].temporal_shifts

# If there is no output_path specified, write with the dataset and SLNet training
if len(args.output_path)==0:
    head, tail = os.path.split(args.checkpoint_SLNet)
    args.output_path = head
# Get commit number 
label = subprocess.check_output(["git", "describe", "--always"]).strip()
save_folder = args.output_path + '/XLFMNet_train__' + datetime.now().strftime('%Y_%m_%d__%H:%M:%S') + '__' + str(label) + '_commit__' + args.prefix


# Get size of the volume
subimage_shape = argsSLNet.subimage_shape

# if args.train_who==2:
    # args.n_frames = 1

dataset = XLFMDatasetFull(args.data_folder, args.lenslet_file, subimage_shape, img_shape=[2160,2160],
            images_to_use=args.images_to_use, divisor=1, isTiff=True, n_frames_net=argsSLNet.n_frames, lenslets_offset=0,
            load_all=True, load_vols=True, load_sparse=True, temporal_shifts=args.temporal_shifts, use_random_shifts=args.use_random_shifts, eval_video=False)


dataset_test = XLFMDatasetFull(args.data_folder_test, args.lenslet_file, subimage_shape, img_shape=[2160,2160],  
            images_to_use=args.images_to_use_test, divisor=1, isTiff=True, n_frames_net=argsSLNet.n_frames, lenslets_offset=0,
            load_all=True, load_vols=True, load_sparse=True, temporal_shifts=args.temporal_shifts, use_random_shifts=args.use_random_shifts, eval_video=False)


n_depths = dataset.get_n_depths()
args.output_shape = subimage_shape + [n_depths]

# Get normalization values 
max_images,max_images_sparse,max_volumes = dataset.get_max() 
if args.use_sparse:
    # Use statistics of sparse images
    mean_imgs,std_images,mean_imgs_sparse,std_images_sparse,mean_vols,std_vols = dataset.get_statistics()
else:
    mean_imgs,std_images,mean_vols,std_vols = dataset.get_statistics()

n_lenslets = dataset.len_lenslets()

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.ceil(args.validation_split * dataset_size))


if args.shuffle_dataset :
    # np.random.seed(261290)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
# Create dataloaders
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

data_loaders = \
    {'train' : \
            data.DataLoader(dataset, batch_size=args.batch_size, 
                                sampler=train_sampler, pin_memory=False, num_workers=n_threads), \
    'val'   : \
            data.DataLoader(dataset, batch_size=args.batch_size,
                                    sampler=valid_sampler, pin_memory=False, num_workers=n_threads), \
    'test'  : \
            data.DataLoader(dataset_test, batch_size=1, pin_memory=False, num_workers=n_threads, shuffle=True)
    }

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Conv3d or type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_uniform(m.weight)
        # torch.nn.init.kaiming_uniform_(m.weight,a=math.sqrt(2))
        # m.weight.data *= 20
        # m.weight.data = m.weight.data.abs()
        # m.bias.data.fill_(0.01)


unet_settings = {'depth':args.unet_depth, 'wf':args.unet_wf, 'drop_out':args.unet_drop_out}

args.unet_settings = unet_settings

# Create net
net = XLFMNet(n_lenslets, args.output_shape, n_temporal_frames=dataset.n_frames, dataset=dataset, use_bias=args.use_bias, unet_settings=unet_settings).to(device)
net.apply(init_weights)

# Trainable parameters
# mean_imgs = mean_imgs_sparse
# std_images = std_images_sparse
trainable_params = [{'params': net.deconv.parameters()}]

params = sum([np.prod(p.size()) for p in net.parameters()])

# Normalization statistics
stats = {'norm_type':args.norm_type, 'norm_type_img':args.norm_type, 'mean_imgs':mean_imgs, 'std_images':std_images, 'max_images':max_images,
        'mean_vols':mean_vols, 'std_vols':std_vols, 'max_vols':max_volumes}



# Create loss function and optimizer
loss = nn.MSELoss()
if args.use_img_loss>0:
    loss_img = nn.MSELoss()

# reuse the gaussian kernel with SSIM & MS_SSIM. 
ssim_module = SSIM(data_range=1, size_average=True, channel=n_lenslets).to(device_repro)

optimizer = torch.optim.Adam(trainable_params, lr=args.learning_rate)

# timers
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# create gradient scaler for mixed precision training
scaler = GradScaler()

start_epoch = 0
if len(args.checkpoint_XLFMNet)>0:
    net.load_state_dict(checkpoint_XLFMNet['model_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint_XLFMNet['optimizer_state_dict'])
    start_epoch = checkpoint_XLFMNet['epoch']-1
    save_folder += '_C'

if len(args.checkpoint_SLNet)>0 and dataset.n_frames>1:
    net.tempConv.load_state_dict(checkpoint_SL['model_state_dict'])
    stats_SLNet = checkpoint_SL['statistics']
    stats['norm_type_img'] = checkpoint_SL['args'].norm_type
    stats['mean_imgs'] = stats_SLNet[0]
    stats['std_images'] = stats_SLNet[1]
else:
    net.tempConv = None


# Create summary writer to log stuff
if debug is False:
    writer = SummaryWriter(log_dir=save_folder)
    writer.add_text('arguments',str(vars(args)),0)
    writer.flush()
    writer.add_scalar('params/', params)

    # Store files
    zf = zipfile.ZipFile(save_folder + "/files.zip", "w")
    for ff in args.files_to_store:
        zf.write(ff)
    zf.close()

import time

if len(args.gpu_repro)>0:
    S = time.time()
    # Load PSF and compute OTF
    n_split = args.n_split
    if debug:
        n_split=60
    OTF,psf_shape = load_PSF_OTF(args.psf_file, args.output_shape, n_depths=n_depths, n_split=n_split, device="cpu")
    OTF = OTF.to(device)
    gc.collect()
    torch.cuda.empty_cache()
    E = time.time()
    print(E - S)

    gc.collect()
    torch.cuda.empty_cache()

    OTF_options =   {'OTF':OTF,
                    'psf_shape':psf_shape,
                    'dataset':dataset,
                    'n_split':n_split,
                    'loss_img':loss_img}

    net.OTF_options = OTF_options

# Update noramlization stats for SLNet inside network
net.stats = stats

if len(args.main_gpu)>1:
    net = nn.DataParallel(net, args.main_gpu, args.main_gpu[0])
    print("Let's use", torch.cuda.device_count(), "GPUs!")

lr = args.learning_rate
# Loop over epochs
for epoch in range(start_epoch, args.max_epochs):
    for curr_train_stage in ['train','val','test']:
        # Grab current data_loader
        curr_loader = data_loaders[curr_train_stage]
        curr_loader_len = curr_loader.sampler.num_samples if curr_train_stage=='test' else len(curr_loader.batch_sampler.sampler.indices)

        if curr_train_stage=='train':
            net.train()
            net.tempConv.eval()
            torch.set_grad_enabled(True)
        if curr_train_stage=='val' or curr_train_stage=='test':
            if epoch%args.eval_every!=0:
                continue
            net.eval()
            torch.set_grad_enabled(False)


        # Store loss
        mean_volume_loss = 0 
        max_grad = 0
        mean_psnr = 0
        mean_time = 0
        mean_repro = 0
        mean_repro_ssim = 0
        # Training
        for ix,(curr_img_stack, local_volumes) in enumerate(curr_loader):

            # If empty or nan in volumes, don't use these for training 
            if curr_img_stack.float().sum()==0 or torch.isnan(curr_img_stack.float().max()):
                continue
            # Normalize volumes if ill posed
            if local_volumes.float().max()>=20000:
                local_volumes = local_volumes.float()
                local_volumes = local_volumes / local_volumes.max() * 4500.0
                local_volumes = local_volumes.half()

            # curr_img_stack returns both the dense and the sparse images, here we only need the sparse.
            if net.tempConv is None:
                assert len(curr_img_stack.shape)>=5, "If sparse is used curr_img_stack should contain both images, dense and sparse stacked in the last dim."
                curr_img_sparse = curr_img_stack[...,-1].clone().to(device) 
                curr_img_stack = curr_img_stack[...,-1].clone().to(device)
            else:
                curr_img_sparse = curr_img_stack[...,-1].clone().to(device)
                curr_img_stack = curr_img_stack[...,0].clone().to(device)
            
            curr_img_stack = curr_img_stack.half()

            curr_img_stack -= args.dark_current
            curr_img_stack = F.relu(curr_img_stack).detach()

            if args.add_noise==1 and curr_train_stage!='test':
                curr_max = curr_img_stack.max()
                # Update new signal power
                signal_power = (args.signal_power_min + (args.signal_power_max-args.signal_power_min) * torch.rand(1)).item()
                curr_img_stack = signal_power/curr_max * curr_img_stack
                # Add noise
                curr_img_stack = pytorch_shot_noise.add_camera_noise(curr_img_stack)
                curr_img_stack = curr_img_stack.float().to(device)

            local_volumes = local_volumes.half().to(device)

            # if conversion to half precission messed up the volumes, continue
            if torch.isinf(local_volumes.max()):
                curr_loader_len -= local_volumes.shape[0]
                continue
              

            # Images are already normalized from mainCreateDataset.py
            # curr_img_stack, local_volumes = normalize_type(curr_img_stack, local_volumes, args.norm_type, mean_imgs, std_images, mean_vols, std_vols, max_images, max_volumes)
            _, local_volumes = normalize_type(curr_img_stack, local_volumes, stats['norm_type'], stats['mean_imgs'], stats['std_images'], stats['mean_vols'], stats['std_vols'], stats['max_images'], stats['max_vols'])
            curr_img_stack, _ = normalize_type(curr_img_stack, local_volumes, stats['norm_type_img'], stats['mean_imgs'], stats['std_images'], stats['mean_vols'], stats['std_vols'], stats['max_images'], stats['max_vols'])
                
            # curr_img_sparse, _ = normalize_type(curr_img_sparse, local_volumes, args.norm_type, mean_imgs_sparse, std_images_sparse, mean_vols, std_vols, max_images, max_volumes)
            

            start.record()

            if curr_train_stage=='train':
                net.zero_grad()
                optimizer.zero_grad()
            # 
            with autocast():
                # Run batch of predicted images in discriminator
                prediction,sparse_prediction = net(curr_img_stack)

                if not all([prediction.shape[i] == subimage_shape[i-2] for i in range(2,4)]):
                    diffY = (subimage_shape[0] - prediction.size()[2])
                    diffX = (subimage_shape[1] - prediction.size()[3])

                    prediction = F.pad(prediction, (diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2))

                # Use only first sparse image, corresponding to the selected volume.
                if net_get_params(net).n_frames>1:
                    curr_img_sparse = curr_img_sparse[:,0,...].unsqueeze(1)
                
                # Extract lenslet images
                curr_img_sparse = dataset.extract_views(curr_img_sparse, dataset.lenslet_coords, dataset.subimage_shape)[:,0,...]

                # curr_img_sparse, _ = normalize_type(curr_img_sparse, local_volumes, args.norm_type, mean_imgs_sparse, std_images_sparse, mean_vols, std_vols, max_images, max_volumes, inverse=True)
            
                volume_loss = loss(local_volumes, prediction)

                if curr_train_stage=='test' and len(args.gpu_repro)>0:
                    with torch.no_grad():
                        reproj_loss, reproj,curr_views,_ = reprojection_loss(sparse_prediction, prediction.float(), OTF, psf_shape, dataset, n_split, device_repro)
                    mean_repro += reproj_loss.item()
                    mean_repro_ssim += ssim_module((sparse_prediction/sparse_prediction.max()).to(device_repro).float(), (reproj/reproj.max()).float().to(device_repro)).cpu().item()
                
            mean_volume_loss += volume_loss.mean().detach().item()

            if curr_train_stage=='train':
                scaler.scale(volume_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                    

            # Record training time
            end.record()
            torch.cuda.synchronize()
            end_time = start.elapsed_time(end)
            mean_time += end_time

            # detach tensors
            local_volumes = local_volumes.detach().cpu().float()
            prediction = prediction.detach().cpu().float()
            curr_img_sparse = curr_img_sparse.detach()
            curr_img_stack = curr_img_stack.detach()

            # Normalize back
            # curr_img_stack, local_volumes = normalize_type(curr_img_stack, local_volumes, args.norm_type, mean_imgs, std_images, mean_vols, std_vols, max_images, max_volumes, inverse=True)
            # _, prediction = normalize_type(curr_img_stack, prediction, args.norm_type, mean_imgs, std_images, mean_vols, std_vols, max_images, max_volumes, inverse=True)
            


            if torch.isinf(torch.tensor(mean_volume_loss)):
                print('inf')
            curr_img_sparse /= curr_img_sparse.max()

            local_volumes -= local_volumes.min()
            prediction -= prediction.min()

            prediction /= max_volumes
            local_volumes /= max_volumes

            curr_img_stack -= curr_img_stack.min()
            curr_img_stack /= curr_img_stack.max()

            # mean_psnr += psnr(local_volumes.detach(), prediction.detach())

        
        mean_volume_loss /= curr_loader_len
        mean_psnr = 20 * torch.log10(max_volumes / torch.sqrt(torch.tensor(mean_volume_loss))) #/= curr_loader_len
        mean_time /= curr_loader_len
        mean_repro /= curr_loader_len
        mean_repro_ssim /= curr_loader_len

        # if epoch%args.eval_every==0:
        #     plt.imshow(volume_2_projections(prediction.permute(0,2,3,1).unsqueeze(1))[0,0,...].float().detach().cpu().numpy())
        #     plt.show()
        if epoch%args.eval_every==0:
            # plot_param_grads(writer, net, epoch, curr_train_stage+'_')

            # plt.subplot(1,3,1)
            # plt.imshow(curr_views[0,10,...].cpu().detach().numpy())
            # plt.subplot(1,3,2)
            # plt.imshow(reproj[0,10,...].cpu().detach().numpy())
            # plt.subplot(1,3,3)
            # plt.imshow((curr_views-reproj)[0,10,...].abs().cpu().detach().float().numpy())
            # plt.title(str(image_loss))
            # plt.show()

            if local_volumes.shape == prediction.shape:
                writer.add_image('max_GT_'+curr_train_stage, tv.utils.make_grid(volume_2_projections(local_volumes.permute(0,2,3,1).unsqueeze(1))[0,...], normalize=True, scale_each=True), epoch)
                writer.add_image('sum_GT_'+curr_train_stage, tv.utils.make_grid(volume_2_projections(local_volumes.permute(0,2,3,1).unsqueeze(1), proj_type=torch.sum)[0,...], normalize=True, scale_each=True), epoch)
            

            writer.add_image('max_prediction_'+curr_train_stage, tv.utils.make_grid(volume_2_projections(prediction.permute(0,2,3,1).unsqueeze(1))[0,...], normalize=True, scale_each=True), epoch)
            writer.add_image('sum_prediction_'+curr_train_stage, tv.utils.make_grid(volume_2_projections(prediction.permute(0,2,3,1).unsqueeze(1), proj_type=torch.sum)[0,...], normalize=True, scale_each=True), epoch)
            
            # input_noisy_grid = tv.utils.make_grid(curr_img_stack[0,0,...].float().unsqueeze(0).cpu().data.detach(), normalize=True, scale_each=False)

            sparse_prediction = sparse_prediction- sparse_prediction.min()
            sparse_prediction /= sparse_prediction.max()
            input_intermediate_sparse_grid = tv.utils.make_grid(sparse_prediction[0,10,...].float().unsqueeze(0).cpu().data.detach(), normalize=True, scale_each=False)
            input_GT_sparse_grid = tv.utils.make_grid(curr_img_sparse[0,10,...].float().unsqueeze(0).cpu().data.detach(), normalize=True, scale_each=False)

            if curr_train_stage=='test' and len(args.gpu_repro)>0:
                repro_grid = tv.utils.make_grid(reproj[0,...].sum(0).float().unsqueeze(0).cpu().data.detach(), normalize=True, scale_each=False)
                writer.add_image('reproj_'+curr_train_stage, repro_grid, epoch)
                repro_grid = tv.utils.make_grid(curr_img_sparse[0,...].sum(0).float().unsqueeze(0).cpu().data.detach(), normalize=True, scale_each=False)
                writer.add_image('reproj_GT_'+curr_train_stage, repro_grid, epoch)
                repro_grid = tv.utils.make_grid((curr_views-reproj)[0,10,...].abs().float().unsqueeze(0).cpu().data.detach(), normalize=True, scale_each=False)
                writer.add_image('reproj_error_'+curr_train_stage, repro_grid, epoch)
                
                writer.add_scalar('reproj/ssim/'+curr_train_stage, mean_repro_ssim, epoch)
                writer.add_scalar('reproj/Loss/'+curr_train_stage, mean_repro, epoch)

            # volGTHist,volPredHist,inputHist = compute_histograms(local_volumes[0,...].float(), prediction[0,...].float(), curr_img_stack[0,...].float())
            
            # writer.add_image('input_noisy_'+curr_train_stage, input_noisy_grid, epoch)
            writer.add_image('image_intermediate_sparse'+curr_train_stage, input_intermediate_sparse_grid, epoch)
            writer.add_image('image_intermediate_sparse_GT'+curr_train_stage, input_GT_sparse_grid, epoch)
            writer.add_scalar('Loss/'+curr_train_stage, mean_volume_loss, epoch)
            writer.add_scalar('psnr/'+curr_train_stage, mean_psnr, epoch)
            writer.add_scalar('times/'+curr_train_stage, mean_time, epoch)
            # writer.add_scalar('grads/'+curr_train_stage, max_grad, epoch)
            writer.add_scalar('lr/'+curr_train_stage, lr, epoch)
            

        # if epoch%2==0:
        print(str(epoch) + ' ' + curr_train_stage + " loss: " + str(mean_volume_loss) + " time: " + str(mean_time))
        if os.path.isfile(main_folder+'exit_file.txt'):
            torch.cuda.empty_cache()
            sys.exit(0)

        if epoch%25==0:
            torch.save({
            'epoch': epoch,
            'args' : args,
            'args_SLNet' : argsSLNet,
            'statistics' : stats,
            'model_state_dict': net_get_params(net).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict' : scaler.state_dict(),
            'loss': mean_volume_loss},
            save_folder + '/model_')#+str(epoch))
        if epoch%50==0:
            torch.save({
            'epoch': epoch,
            'args' : args,
            'args_SLNet' : argsSLNet,
            'statistics' : stats,
            'model_state_dict': net_get_params(net).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict' : scaler.state_dict(),
            'loss': mean_volume_loss},
            save_folder + '/model_'+str(epoch))

    

