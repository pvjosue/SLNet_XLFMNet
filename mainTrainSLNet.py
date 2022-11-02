import torch
import sys
from PIL import Image

from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler,SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast,GradScaler
import torchvision as tv
import torch.nn as nn
import matplotlib.pyplot as plt
import subprocess
import numpy as np
from datetime import datetime
import argparse
import math
import zipfile
from tqdm import tqdm
import utils.pytorch_shot_noise as pytorch_shot_noise
from nets.SLNet import SLNet
from utils.XLFMDataset import XLFMDatasetFull
from utils.misc_utils import *


runs_dir = ""
data_dir = ""
main_folder = "XLFMNet/"
runs_dir = "XLFMNet/runs/"
data_dir = "XLFM/"
# Real image 

dataset_paths = {   
                    'fish2'     : f'{data_dir}/dataset_fish2_10Hz/',
                    'fish_conf' : f'{data_dir}/fish_conf/',
                }

dataset_to_use = 'fish_conf'
dataset_to_use_test = 'fish2'

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', nargs='?', default= dataset_paths[dataset_to_use], help='Input training images path in format /XLFM_image/XLFM_image_stack.tif and XLFM_image_stack_S.tif in case of a sparse GT stack.')
parser.add_argument('--data_folder_test', nargs='?', default= dataset_paths[dataset_to_use_test], help='Input testing image path')
parser.add_argument('--lenslet_file', nargs='?', default= "lenslet_coords.txt", help='Text file with the lenslet coordinates pairs x y "\n"')

parser.add_argument('--files_to_store', nargs='+', default=[], help='Relative paths of files to store in a zip when running this script, for backup.')
parser.add_argument('--prefix', nargs='?', default= "Fish2", help='Prefix string for the output folder.')
parser.add_argument('--checkpoint', nargs='?', default= "", help='File path of checkpoint of previous run.')
# Images related arguments
parser.add_argument('--images_to_use', nargs='+', type=int, default=list(range(0,90,1)), help='Indeces of images to train on.')
parser.add_argument('--images_to_use_test', nargs='+', type=int, default=list(range(0,90,1)), help='Indeces of images to test on.')
parser.add_argument('--lenslet_crop_size', type=int, default=512, help='Side size of the microlens image.')
parser.add_argument('--img_size', type=int, default=2160, help='Side size of input image, square prefered.')
# Training arguments
parser.add_argument('--batch_size', type=int, default=8, help='Training batch size.') 
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Training learning rate.')
parser.add_argument('--max_epochs', type=int, default=101, help='Training epochs to run.')
parser.add_argument('--validation_split', type=float, default=0.1, help='Which part to use for validation 0 to 1.')
parser.add_argument('--eval_every', type=int, default=10, help='How often to evaluate the testing/validaton set.')
parser.add_argument('--shuffle_dataset', type=int, default=1, help='Radomize training images 0 or 1')
parser.add_argument('--use_bias', type=int, default=0, help='Use bias during training? 0 or 1')
parser.add_argument('--plot_images', type=int, default=0, help='Plot results with matplotlib?')
# Noise arguments
parser.add_argument('--add_noise', type=int, default=0, help='Apply noise to images? 0 or 1')
parser.add_argument('--signal_power_max', type=float, default=30**2, help='Max signal value to control signal to noise ratio when applyting noise.')
parser.add_argument('--signal_power_min', type=float, default=60**2, help='Min signal value to control signal to noise ratio when applyting noise.')
parser.add_argument('--norm_type', type=float, default=2, help='Normalization type, see the normalize_type function for more info.')
parser.add_argument('--dark_current', type=float, default=106, help='Dark current value of camera.')
parser.add_argument('--dark_current_sparse', type=float, default=0, help='Dark current value of camera.')
# Sparse decomposition arguments
parser.add_argument('--n_frames', type=int, default=3, help='Number of frames used as input to the SLNet.')
parser.add_argument('--rank', type=int, default=3, help='Rank enforcement for SVD. 6 is good')
parser.add_argument('--SL_alpha_l1', type=float, default=0.1, help='Threshold value for alpha in sparse decomposition.')
parser.add_argument('--SL_mu_sum_constraint', type=float, default=1e-2, help='Threshold value for mu in sparse decomposition.')
parser.add_argument('--weight_multiplier', type=float, default=0.5, help='Initialization multiplyier for weights, important parameter.')
# SLNet config
parser.add_argument('--temporal_shifts', nargs='+', type=int, default=[0,49,99], help='Which frames to use for training and testing.')
parser.add_argument('--use_random_shifts', nargs='+', type=int, default=0, help='Randomize the temporal shifts to use? 0 or 1')
parser.add_argument('--frame_to_grab', type=int, default=0, help='Which frame to show from the sparse decomposition?')
parser.add_argument('--l0_ths', type=float, default=0.05, help='Threshold value for alpha in nuclear decomposition')
# misc arguments
parser.add_argument('--output_path', nargs='?', default=runs_dir + '/camera_ready_github/')
parser.add_argument('--main_gpu', nargs='+', type=int, default=[], help='List of GPUs to use: [0,1]')

n_threads = 0
args = parser.parse_args()
if len(args.main_gpu)>0:
    device = "cuda:" + str(args.main_gpu[0])
else:
    device = "cuda"
    args.main_gpu = [0]

if n_threads!=0:
    torch.set_num_threads(n_threads)

checkpoint_path = None
if len(args.checkpoint)>0:
    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_path = args.checkpoint
    currArgs = args
    args = checkpoint['args']
    args.max_epochs = currArgs.max_epochs
    args.images_to_use = currArgs.images_to_use
    args.dark_current = currArgs.dark_current
    args.learning_rate = currArgs.learning_rate
    args.batch_size = currArgs.batch_size
    args.data_folder_test = currArgs.data_folder_test
    args.dark_current_sparse = currArgs.dark_current_sparse
args.shuffle_dataset = bool(args.shuffle_dataset)


# Get commit number 
label = subprocess.check_output(["git", "describe", "--always"]).strip()
save_folder = args.output_path + datetime.now().strftime('%Y_%m_%d__%H:%M:%S') + str(args.main_gpu[0]) + "_gpu__" + args.prefix

print(f'Logging dir: {save_folder}')

# Load datasets
args.subimage_shape = 2*[args.lenslet_crop_size]
args.output_shape = 2*[args.lenslet_crop_size]
dataset = XLFMDatasetFull(args.data_folder, args.lenslet_file, args.subimage_shape, img_shape=2*[args.img_size],
            images_to_use=args.images_to_use,
            load_sparse=False, load_vols=False, temporal_shifts=args.temporal_shifts, use_random_shifts=args.use_random_shifts)


dataset_test = XLFMDatasetFull(args.data_folder_test, args.lenslet_file, args.subimage_shape, 2*[args.img_size],  
            images_to_use=args.images_to_use_test,
            load_vols=False, load_sparse=False)

# Get normalization values 
max_images,max_images_sparse,max_volumes = dataset.get_max() 
mean_imgs,std_images,mean_vols,std_vols = dataset.get_statistics()


dataset.stacked_views, _ = normalize_type(dataset.stacked_views, 0, args.norm_type, mean_imgs, std_images, mean_vols, std_vols, max_images, max_volumes)
dataset_test.stacked_views, _ = normalize_type(dataset_test.stacked_views, 0, args.norm_type, mean_imgs, std_images, mean_vols, std_vols, max_images, max_volumes)

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.ceil(args.validation_split * dataset_size))

torch.manual_seed(261290)

if args.shuffle_dataset :
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

# Eval samples
data_loaders_save = \
    {'train' : \
            data.DataLoader(dataset, batch_size=1, 
                                sampler=SequentialSampler(list(range(dataset_size))), pin_memory=False, num_workers=n_threads), \
    'test'  : \
            data.DataLoader(dataset_test, batch_size=1, 
                                sampler=SequentialSampler(list(range(len(dataset_test)))), pin_memory=False, num_workers=n_threads, shuffle=False)
    }


# Weight initialization function
def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Conv3d or type(m) == nn.ConvTranspose2d:
        torch.nn.init.kaiming_uniform_(m.weight,a=math.sqrt(2))
        m.weight.data = m.weight.data.abs()*args.weight_multiplier


# Create net
net = SLNet(dataset.n_frames, use_bias=args.use_bias, mu_sum_constraint=args.SL_mu_sum_constraint, alpha_l1=args.SL_alpha_l1).to(device)
net.apply(init_weights)

# Use multiple gpus?
if len(args.main_gpu)>1:
    net = nn.DataParallel(net, args.main_gpu, args.main_gpu[0])
    print("Let's use", torch.cuda.device_count(), "GPUs!")

# Trainable parameters
trainable_params = list(net.parameters())
params = sum([np.prod(p.size()) for p in net.parameters()])

# Create optimizer
optimizer = torch.optim.Adam(trainable_params, lr=args.learning_rate)

# create gradient scaler for mixed precision training
scaler = GradScaler()

# Is there a checkpoint? load it
start_epoch = 0
if checkpoint_path:
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch']-1
    save_folder += '_C'

# Create summary writer to log stuff
writer = SummaryWriter(log_dir=save_folder)
writer.add_text('arguments',str(vars(args)),0)
writer.flush()
writer.add_scalar('params/', params)

# Store files for backup
zf = zipfile.ZipFile(save_folder + "/files.zip", "w")
for ff in args.files_to_store:
    zf.write(ff)
zf.close()

# timers
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# Loop over epochs
for epoch in range(start_epoch, args.max_epochs):
    for curr_train_stage in ['train','val','test']:
        # Grab current data_loader
        curr_loader = data_loaders[curr_train_stage]
        curr_loader_len = curr_loader.sampler.num_samples if curr_train_stage=='test' else len(curr_loader.batch_sampler.sampler.indices)

        if curr_train_stage=='train':
            net.train()
            torch.set_grad_enabled(True)
        if curr_train_stage=='val' or curr_train_stage=='test':
            if epoch%args.eval_every!=0:
                continue
            net.eval()
            torch.set_grad_enabled(False)


        # Store losses of current epoch
        mean_loss = 0 
        mean_psnr = 0
        mean_time = 0
        mean_eigen_values = torch.zeros([args.n_frames])
        mean_eigen_values_cropped = torch.zeros([args.n_frames])
        mean_eigen_crop = 0

        perf_metrics = {}
        perf_metrics['Rank_SLNet'] = []
        perf_metrics['Fro_SLNet'] = []
        perf_metrics['Fro_Ratio_SLNet'] = []
        perf_metrics['mean_error_SLNet'] = []
        perf_metrics['L1_SLNet'] = []

        # Training
        for ix,(curr_img_stack, _) in enumerate(tqdm(curr_loader, desc='Optimizing images')):
            curr_img_stack = curr_img_stack.to(device)
            # Apply noise if needed, and only in the test set, as the train set comes from real images
            if args.add_noise==1 and curr_train_stage!='test':
                curr_max = curr_img_stack.max()
                # Update new signal power
                signal_power = (args.signal_power_min + (args.signal_power_max-args.signal_power_min) * torch.rand(1)).item()
                curr_img_stack = signal_power/curr_max * curr_img_stack
                # Add noise
                curr_img_stack = pytorch_shot_noise.add_camera_noise(curr_img_stack)
                curr_img_stack = curr_img_stack.to(device)

                
            if curr_train_stage=='train':
                net.zero_grad()
                optimizer.zero_grad()

            with autocast():
                torch.cuda.synchronize()
                start.record()
                # Predict dense part with the network
                dense_part = F.relu(net(curr_img_stack))

                # Compute sparse part
                sparse_part = F.relu(curr_img_stack-dense_part)

                # Measure time
                end.record()
                torch.cuda.synchronize()
                end_time = start.elapsed_time(end) / curr_img_stack.shape[0]
                mean_time += end_time

                # Compute sparse decomposition on a patch, as the full image doesn't fit in memory due to SVD
                center = 64
                if curr_train_stage!='train':
                    center = 32
                coord_to_crop = torch.randint(center,dense_part.shape[3]-center, [2])
                
                # Grab patches
                dense_crop = dense_part[:,:,coord_to_crop[0]-center:coord_to_crop[0]+center,coord_to_crop[1]-center:coord_to_crop[1]+center].contiguous()
                sparse_crop = sparse_part[:,:,coord_to_crop[0]-center:coord_to_crop[0]+center,coord_to_crop[1]-center:coord_to_crop[1]+center].contiguous()
                curr_img_crop = curr_img_stack[:,:,coord_to_crop[0]-center:coord_to_crop[0]+center,coord_to_crop[1]-center:coord_to_crop[1]+center].detach()
                
                # Reconstruction error
                Y = (curr_img_crop - dense_crop - sparse_crop)
                # Nuclear norm
                dense_vector = dense_crop.view(dense_part.shape[0],dense_part.shape[1],-1)
                with autocast(enabled=False):
                    (u,s,v) = torch.svd_lowrank(dense_vector.permute(0,2,1).float(), q=args.rank)
                    # eigenvalues thresholding operation
                    s = torch.sign(s) * torch.max(s.abs() - net.mu_sum_constraint, torch.zeros_like(s))

                # Reconstruct the images from the eigen information
                reconstructed_vector = torch.zeros([dense_crop.shape[0],dense_crop.shape[1],dense_crop.shape[2]*dense_crop.shape[3]],device=device)
                for nB in range(s.shape[0]):
                    currS = torch.diag(s[nB,:])
                    reconstructed_vector[nB,...] = torch.mm(torch.mm(u[nB,...], currS), v[nB,...].t()).t()
                reconstructed_dense = reconstructed_vector.view(dense_crop.shape)

                # Compute full loss
                full_loss = F.l1_loss(reconstructed_dense,curr_img_crop) + net.alpha_l1 * sparse_crop.abs().mean() + Y.abs().mean()

                
                
                if ix==0 and args.plot_images:
                    sparse_crop = F.relu(curr_img_crop - reconstructed_dense)
                    plt.clf()
                    for n in range(0,3):
                        plt.subplot(3,4,4*n+1)
                        plt.imshow(curr_img_crop[0,n,...].detach().cpu().float().numpy())
                        plt.title('Input')
                        plt.subplot(3,4,4*n+2)
                        plt.imshow(dense_crop[0,n,...].detach().cpu().float().numpy())
                        plt.title('Dense prediction')
                        plt.subplot(3,4,4*n+3)
                        plt.imshow(sparse_crop[0,n,...].detach().cpu().float().numpy())
                        plt.title('Sparse prediction')
                        plt.subplot(3,4,4*n+4)
                        plt.imshow(Y[0,n,...].detach().cpu().float().numpy())
                        plt.title('Y')
                    plt.pause(0.1)       
                    plt.draw()


            if curr_train_stage=='train':
                full_loss.backward()

                # Check fo NAN in training
                broken = False
                with torch.no_grad():
                    for param in net.parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad.mean()):
                                broken = True
                if broken:
                    continue

                optimizer.step()


            # detach tensors for display
            # curr_img_sparse = curr_img_sparse.detach()
            curr_img_stack = curr_img_stack.detach()
            dense_part = dense_part.detach()
            sparse_part = sparse_part.detach()

            # Normalize back
            curr_img_stack,_ = normalize_type(curr_img_stack.float(), 0, args.norm_type, mean_imgs, std_images, mean_vols, std_vols, max_images, max_volumes, inverse=True)
            sparse_part,_ = normalize_type(sparse_part.float(), 0, args.norm_type, mean_imgs, std_images, mean_vols, std_vols, max_images, max_volumes, inverse=True)
            dense_part,_ = normalize_type(dense_part.float(), 0, args.norm_type, mean_imgs, std_images, mean_vols, std_vols, max_images, max_volumes, inverse=True)
            
            sparse_part = F.relu(curr_img_stack-dense_part.detach())
            mean_loss += full_loss.item()

        # Compute different performance metrics
        mean_loss /= curr_loader_len
        mean_psnr = 20 * torch.log10(max_images / torch.sqrt(torch.tensor(mean_loss))) 
        mean_time /= curr_loader_len


        if epoch%args.eval_every==0:
            # Create debug images
            M = curr_img_stack[:,args.frame_to_grab,...].unsqueeze(1)
            S_SLNet = sparse_part[:,args.frame_to_grab,...].unsqueeze(1)
            L_SLNet = dense_part[:,args.frame_to_grab,...].unsqueeze(1)
            Rank_SLNet = torch.matrix_rank(L_SLNet[0,0,...].float()).item()

            fro_M = torch.norm(M).item()
            fro_SLNet = torch.norm(M-L_SLNet-S_SLNet).item()
            mean_error = (M-L_SLNet-S_SLNet).mean().item()
            L1_SLNet = (S_SLNet>(args.l0_ths*S_SLNet.max())).float().sum().item() / torch.numel(S_SLNet)

            perf_metrics['L1_SLNet'].append(L1_SLNet)
            perf_metrics['mean_error_SLNet'].append(mean_error)
            perf_metrics['Rank_SLNet'].append(Rank_SLNet)
            perf_metrics['Fro_SLNet'].append(fro_SLNet)
            perf_metrics['Fro_Ratio_SLNet'].append(fro_SLNet/fro_M)

            
            input_noisy_grid = tv.utils.make_grid(curr_img_stack[0,0,...].float().unsqueeze(0).cpu().data.detach(), normalize=True, scale_each=False)

            sparse_part = F.relu(sparse_part.detach()).float()
            dense_prediction = F.relu(dense_part.detach()).float()
            reconstructed_dense_prediciton = F.relu(reconstructed_dense.detach()).float()

            Y = sparse_part+dense_prediction

            sparse_part /= Y.max()
            input_intermediate_sparse_grid = tv.utils.make_grid(sparse_part[0,0,...].float().unsqueeze(0).cpu().data.detach(), normalize=True, scale_each=False)
            
            dense_prediction /= Y.max()
            input_intermediate_dense_grid = tv.utils.make_grid(dense_prediction[0,0,...].float().unsqueeze(0).cpu().data.detach(), normalize=True, scale_each=False)
          
            writer.add_image('input/'+curr_train_stage, input_noisy_grid, epoch)
            writer.add_image('sparse/'+curr_train_stage, input_intermediate_sparse_grid, epoch)
            writer.add_image('dense/'+curr_train_stage, input_intermediate_dense_grid, epoch)
            writer.add_scalar('Loss/'+curr_train_stage, mean_loss, epoch)
            # writer.add_scalar('Loss/mean_sparse_l1_'+curr_train_stage, mean_sparse_l1, epoch)
            writer.add_scalar('regularization_weights/alpha_l1', net.alpha_l1, epoch)
            writer.add_scalar('regularization_weights/mu_sum_constraint', net.mu_sum_constraint.item(), epoch)
            writer.add_scalar('regularization_weights/eigen_crop_percentage', mean_eigen_crop, epoch)
            writer.add_scalar('psnr/'+curr_train_stage, mean_psnr, epoch)
            writer.add_scalar('times/'+curr_train_stage, mean_time, epoch)
            writer.add_scalar('lr/'+curr_train_stage, args.learning_rate, epoch)
            
            # writer.add_histogram('eigenvalues/'+curr_train_stage, mean_eigen_values, epoch)
            # writer.add_histogram('eigenvalues_cropped/'+curr_train_stage, mean_eigen_values_cropped, epoch)


            for k,v in perf_metrics.items():
                writer.add_scalar('metrics/'+k+'_'+curr_train_stage, v[-1], epoch)

        print(str(epoch) + ' ' + curr_train_stage + " loss: " + str(mean_loss) + " eigenCrop: " + str(mean_eigen_crop) + " time: " + str(mean_time))#, end="\r")

        if epoch%25==0:
            torch.save({
            'epoch': epoch,
            'args' : args,
            'statistics' : [mean_imgs,std_images,mean_vols,std_vols ],
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict' : scaler.state_dict(),
            'loss': mean_loss},
            save_folder + '/model_'+str(epoch))
            
            with torch.no_grad():
                for curr_train_stage in ['train']:
                    curr_loader = data_loaders_save[curr_train_stage]
                    output_sparse_images = torch.zeros_like(curr_img_stack[0,0,...].unsqueeze(0).unsqueeze(0), device='cpu').repeat(len(curr_loader),1,1,1)
                    for ix,(curr_img_stack, _) in enumerate(curr_loader):      
                        curr_img_stack = curr_img_stack.to(device)      
                        with autocast():
                            # Predict dense part with the network
                            dense_part = F.relu(net(curr_img_stack))

                            # Compute sparse part
                            sparse_part = F.relu(curr_img_stack-dense_part)
                            output_sparse_images[ix,...] = sparse_part[0,0,].detach().cpu()
                    save_image(output_sparse_images.permute(1,0,2,3),f'{save_folder}/Sparse_{curr_train_stage}_ep_{epoch}.tif')
