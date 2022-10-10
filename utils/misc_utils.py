import torch
import torchvision as tv
import torch.nn.functional as F
from waveblocks.utils import complex_operations as ob
from PIL import Image
import torchvision.transforms as TF
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms
from scipy.ndimage.filters import gaussian_filter
import h5py
import gc
import re
import numpy as np
import findpeaks
import pickle
from tifffile import imsave

# Prepare a volume to be shown in tensorboard as an image
def volume_2_tensorboard(vol, batch_index=0, z_scaling=2):
    vol = vol.detach()
    # expecting dims to be [batch, depth, xDim, yDim]
    xyProj = tv.utils.make_grid(vol[batch_index,...].float().unsqueeze(0).sum(1).cpu().data, normalize=True, scale_each=True)
    
    # interpolate z in case that there are not many depths
    vol = torch.nn.functional.interpolate(vol.permute(0,2,3,1).unsqueeze(1), (vol.shape[2], vol.shape[3], vol.shape[1]*z_scaling))
    yzProj = tv.utils.make_grid(vol[batch_index,...].float().unsqueeze(0).sum(3).cpu().data, normalize=True, scale_each=True)
    xzProj = tv.utils.make_grid(vol[batch_index,...].float().unsqueeze(0).sum(2).cpu().data, normalize=True, scale_each=True)

    return xzProj, yzProj, xyProj

# Convert volume to single 2D MIP image, input [batch,1,xDim,yDim,zDim]
def volume_2_projections(vol, proj_type=torch.max):
    # vol = vol.detach()
    vol_size = vol.shape
    if proj_type is torch.max or proj_type is torch.min:
        x_projection,_ = proj_type(vol.float().cpu(), dim=2)
        y_projection,_ = proj_type(vol.float().cpu(), dim=3)
        z_projection,_ = proj_type(vol.float().cpu(), dim=4)
    elif proj_type is torch.sum:
        x_projection = proj_type(vol.float().cpu(), dim=2)
        y_projection = proj_type(vol.float().cpu(), dim=3)
        z_projection = proj_type(vol.float().cpu(), dim=4)

    out_img = torch.zeros(vol_size[0], vol_size[1], vol_size[2] + vol_size[4], vol_size[3] + vol_size[4])

    out_img[:,:,:vol_size[2], :vol_size[3]] = z_projection
    out_img[:,:,vol_size[2]:, :vol_size[3]] = x_projection.permute(0,1,3,2)
    out_img[:,:,:vol_size[2], vol_size[3]:] = y_projection

    # Draw white lines
    out_img[:,:,vol_size[2],...] = z_projection.max()
    out_img[:,:,:,vol_size[3],...] = z_projection.max()

    return out_img


# Aid functions for shiftfft2
def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)
def batch_fftshift2d_real(x):
    out = x
    for dim in range(2, len(out.size())):
        n_shift = x.size(dim)//2
        if x.size(dim) % 2 != 0:
            n_shift += 1  # for odd-sized images
        out = roll_n(out, axis=dim, n=n_shift)
    return out  

# FFT convolution, the kernel fft can be precomputed
def fft_conv(A,B, fullSize, Bshape=[],B_precomputed=False):
    import torch.fft
    nDims = A.ndim-2
    # fullSize = torch.tensor(A.shape[2:]) + Bshape
    # fullSize = torch.pow(2, torch.ceil(torch.log(fullSize.float())/torch.log(torch.tensor(2.0)))-1)
    padSizeA = (fullSize - torch.tensor(A.shape[2:]))
    padSizesA = torch.zeros(2*nDims,dtype=int)
    padSizesA[0::2] = torch.floor(padSizeA/2.0)
    padSizesA[1::2] = torch.ceil(padSizeA/2.0)
    padSizesA = list(padSizesA.numpy()[::-1])

    A_padded = F.pad(A,padSizesA)
    Afft = torch.fft.rfft2(A_padded)
    if B_precomputed:
        return batch_fftshift2d_real(torch.fft.irfft2( Afft * B.detach()))
    else:
        padSizeB = (fullSize - torch.tensor(B.shape[2:]))
        padSizesB = torch.zeros(2*nDims,dtype=int)
        padSizesB[0::2] = torch.floor(padSizeB/2.0)
        padSizesB[1::2] = torch.ceil(padSizeB/2.0)
        padSizesB = list(padSizesB.numpy()[::-1])
        B_padded = F.pad(B,padSizesB)
        Bfft = torch.fft.rfft2(B_padded)
        return batch_fftshift2d_real(torch.fft.irfft2( Afft * Bfft.detach())), Bfft.detach()


def reprojection_loss_camera(gt_imgs, prediction, PSF, camera, dataset, device="cpu"):
    out_type = gt_imgs.type()
    camera = camera.to(device)
    reprojection = camera(prediction.to(device), PSF.to(device))
    reprojection_views = dataset.extract_views(reprojection, dataset.lenslet_coords, dataset.subimage_shape)[0,0,...]
    loss = F.mse_loss(gt_imgs.float().to(device), reprojection_views.float().to(device))

    return loss.type(out_type), reprojection_views.type(out_type), gt_imgs.type(out_type), reprojection.type(out_type)

def reprojection_loss(gt_imgs, prediction, OTF, psf_shape, dataset, n_split=20, device="cpu", loss=F.mse_loss):
    out_type = gt_imgs.type()
    batch_size = prediction.shape[0]
    reprojection = fft_conv_split(prediction[0,...].unsqueeze(0), OTF, psf_shape, n_split, B_precomputed=True, device=device)

    reprojection_views = torch.zeros_like(gt_imgs)
    reprojection_views[0,...] = dataset.extract_views(reprojection, dataset.lenslet_coords, dataset.subimage_shape)[0,0,...]

    # full_reprojection = reprojection.detach()
    # reprojection_views = reprojection_views.unsqueeze(0).repeat(batch_size,1,1,1)
    for nSample in range(1,batch_size):
        reprojection = fft_conv_split(prediction[nSample,...].unsqueeze(0), OTF, psf_shape, n_split, B_precomputed=True, device=device)
        reprojection_views[nSample,...] = dataset.extract_views(reprojection, dataset.lenslet_coords, dataset.subimage_shape)[0,0,...]
        # full_reprojection += reprojection.detach()

    # gt_imgs /= gt_imgs.float().max()
    # reprojection_views /= reprojection_views.float().max()
    # loss = F.mse_loss(gt_imgs[gt_imgs!=0].to(device), reprojection_views[gt_imgs!=0])
    #loss = (1-gt_imgs[reprojection_views!=0]/reprojection_views[reprojection_views!=0]).abs().mean()
    loss = loss(gt_imgs.float().to(device), reprojection_views.float().to(device))

    return loss.type(out_type), reprojection_views.type(out_type), gt_imgs.type(out_type), reprojection.type(out_type)

# Split an fft convolution into batches containing different depths
def fft_conv_split(A, B, psf_shape, n_split, B_precomputed=False, device = "cpu"):
    n_depths = A.shape[1]
    
    split_conv = n_depths//n_split
    depths = list(range(n_depths))
    depths = [depths[i:i + split_conv] for i in range(0, n_depths, split_conv)]

    fullSize = torch.tensor(A.shape[2:]) + psf_shape
    
    crop_pad = [(psf_shape[i] - fullSize[i])//2 for i in range(0,2)]
    crop_pad = (crop_pad[1], (psf_shape[-1]- fullSize[-1])-crop_pad[1], crop_pad[0], (psf_shape[-2] - fullSize[-2])-crop_pad[0])
    # Crop convolved image to match size of PSF
    img_new = torch.zeros(A.shape[0], 1, psf_shape[0], psf_shape[1], device=device)
    if B_precomputed == False:
        OTF_out = torch.zeros(1, n_depths, fullSize[0], fullSize[1]//2+1, requires_grad=False, dtype=torch.complex64, device=device)
    for n in range(n_split):
        # print(n)
        curr_psf = B[:,depths[n],...].to(device)
        img_curr = fft_conv(A[:,depths[n],...].to(device), curr_psf, fullSize, psf_shape, B_precomputed)
        if B_precomputed == False:
            OTF_out[:,depths[n],...] = img_curr[1]
            img_curr = img_curr[0]
        img_curr = F.pad(img_curr, crop_pad)
        img_new += img_curr[:,:,:psf_shape[0],:psf_shape[1]].sum(1).unsqueeze(1).abs()
    
    if B_precomputed == False:
        return img_new, OTF_out
    return img_new


def imadjust(x,a,b,c,d,gamma=1):
    # Similar to imadjust in MATLAB.
    # Converts an image range from [a,b] to [c,d].
    # The Equation of a line can be used for this transformation:
    #   y=((d-c)/(b-a))*(x-a)+c
    # However, it is better to use a more generalized equation:
    #   y=((x-a)/(b-a))^gamma*(d-c)+c
    # If gamma is equal to 1, then the line equation is used.
    # When gamma is not equal to 1, then the transformation is not linear.

    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    mask = (y>0).float()
    y = torch.mul(y,mask)
    return y


# Apply different normalizations to volumes and images
def normalize_type(LF_views, vols, id=0, mean_imgs=0, std_imgs=1, mean_vols=0, std_vols=1, max_imgs=1, max_vols=1, inverse=False):
    if inverse:
        if id==-1: # No normalization
            return LF_views, vols
        if id==0: # baseline normlization
            return (LF_views) * (2*std_imgs), vols * std_vols + mean_vols
        if id==1: # Standarization of images and volume normalization
            return LF_views * std_imgs + mean_imgs, vols * std_vols
        if id==2: # normalization of both
            return LF_views * max_imgs, vols * max_vols
        if id==3: # normalization of both
            return LF_views * std_imgs, vols * std_vols
    else:
        if id==-1: # No normalization
            return LF_views, vols
        if id==0: # baseline normlization
            return (LF_views) / (2*std_imgs), (vols - mean_vols) / std_vols
        if id==1: # Standarization of images and volume normalization
            return (LF_views - mean_imgs) / std_imgs, vols / std_vols
        if id==2: # normalization of both
            return LF_views / max_imgs, vols / max_vols
        if id==3: # normalization of both
            return LF_views / std_imgs, vols / std_vols


# Random transformation of volume, for augmentation
def transform_volume(currVol, transformParams=None, maxZRoll=180):
    # vol format [B,Z,X,Y]
    if transformParams==None:
        angle, transl, scale, shear = TF.RandomAffine.get_params((-180,180), (0.1,0.1), (0.9,1.1), (0,0), currVol.shape[2:4])
        zRoll = int(maxZRoll*torch.rand(1)-maxZRoll//2)
        transformParams = {'angle':angle, 'transl':transl, 'scale':scale, 'shear':shear, 'zRoll':zRoll}
    
    zRoll = transformParams['zRoll']
    for nVol in range(currVol.shape[0]):
        for nDepth in range(currVol.shape[1]):
            currDepth = TF.functional.to_pil_image(currVol[nVol,nDepth,...].float())
            currDepth = TF.functional.affine(currDepth, transformParams['angle'], transformParams['transl'], transformParams['scale'], transformParams['shear'])
            currVol[nVol,nDepth,...] = TF.functional.to_tensor(currDepth)
    currVol = currVol.roll(zRoll, 1)
    if zRoll>=0:
        currVol[:,0:zRoll,...] = 0
    else:
        currVol[:,zRoll:,...] = 0
    return currVol, transformParams

def plot_param_grads(writer, net, curr_it, prefix=""):
    for tag, parm in net.named_parameters():
        if parm.grad is not None:
            writer.add_histogram(prefix+tag, parm.grad.data.cpu().numpy(), curr_it)
            assert not torch.isnan(parm.grad.sum()), print("NAN in: " + str(tag) + "\t\t")

def compute_histograms(gt, pred, input_img, n_bins=1000):
    volGTHist = torch.histc(gt, bins=n_bins, max=gt.max().item())
    volPredHist = torch.histc(pred, bins=n_bins, max=pred.max().item())
    inputHist = torch.histc(input_img, bins=n_bins, max=input_img.max().item())
    return volGTHist,volPredHist,inputHist


def match_histogram(source, reference):
    isTorch = False
    source = source / source.max() * reference.max()
    if isinstance(source, torch.Tensor):
        source = source.cpu().numpy()
        isTorch = True
    if isinstance(reference, torch.Tensor):
        reference = reference[:source.shape[0],...].cpu().numpy()

    matched = match_histograms(source, reference, multichannel=False)
    if isTorch:
        matched = torch.from_numpy(matched)
    return matched

def load_PSF(filename, n_depths=120):
    # Load PSF
    try:
        # Check permute
        psfIn = torch.from_numpy(loadmat(filename)['PSF']).permute(2,0,1).unsqueeze(0)
    except:
        psfFile = h5py.File(filename,'r')
        psfIn = torch.from_numpy(psfFile.get('PSF')[:]).permute(0,2,1).unsqueeze(0)

    # Make a square PSF
    min_psf_size = min(psfIn.shape[-2:])
    psf_pad = [min_psf_size-psfIn.shape[-1], min_psf_size-psfIn.shape[-2]]
    psf_pad = [psf_pad[0]//2, psf_pad[0]//2, psf_pad[1],psf_pad[1]]
    psfIn = F.pad(psfIn, psf_pad)

    # Grab only needed depths
    psfIn = psfIn[:, psfIn.shape[1]//2- n_depths//2+1 : psfIn.shape[1]//2+n_depths//2+1, ...]
    # Normalize psfIn such that each depth sum is equal to 1
    for nD in range(psfIn.shape[1]):
        psfIn[:,nD,...] = psfIn[:,nD,...] / psfIn[:,nD,...].sum()
    
    return psfIn

def load_PSF_OTF(filename, vol_size, n_split=20, n_depths=120, downS=1, device="cpu",
                 dark_current=106, calc_max=False, psfIn=None, compute_transpose=False,
                 n_lenslets=29, lenslet_centers_file_out='lenslet_centers_python.txt'):
    # Load PSF
    if psfIn is None:
        psfIn = load_PSF(filename, n_depths)

    if len(lenslet_centers_file_out)>0:
        find_lenslet_centers(psfIn[0,n_depths//2,...].numpy(), n_lenslets=n_lenslets, file_out_name=lenslet_centers_file_out)
    if calc_max:
        psfMaxCoeffs = torch.amax(psfIn, dim=[0,2,3])

    psf_shape = torch.tensor(psfIn.shape[2:])
    vol = torch.rand(1,psfIn.shape[1], vol_size[0], vol_size[1], device=device)
    img, OTF = fft_conv_split(vol, psfIn.float().detach().to(device), psf_shape, n_split=n_split, device=device)
    
    OTF = OTF.detach()

    if compute_transpose:
        OTFt = torch.real(OTF) - 1j * torch.imag(OTF)
        OTF = torch.cat((OTF.unsqueeze(-1), OTFt.unsqueeze(-1)), 4)
    if calc_max:
        return OTF, psf_shape, psfMaxCoeffs
    else:
        return OTF,psf_shape


def find_lenslet_centers(img, n_lenslets=29, file_out_name='lenslet_centers_python.txt'):
    fp2 = findpeaks.findpeaks()
    
    image_divisor = 4 # To find the centers faster
    img = findpeaks.stats.resize(img, size=(img.shape[0]//image_divisor,img.shape[1]//image_divisor))
    results_2 = fp2.fit(img)
    limit_min = fp2.results['persistence'][0:n_lenslets+1]['score'].min()

    # Initialize topology
    fp = findpeaks.findpeaks(method='topology', limit=limit_min)
    # make the fit
    results = fp.fit(img)
    # Make plot
    fp.plot_persistence()
    # fp.plot()
    results = np.ndarray([n_lenslets,2], dtype=int)
    for ix,data in enumerate(fp.results['groups0']):
        results[ix] = np.array(data[0], dtype=int) * image_divisor
    if len(file_out_name) > 0:
        np.savetxt(file_out_name, results, fmt='%d', delimiter='\t')

    return results

# Aid functions for getting information out of directory names
def get_intensity_scale_from_name(name):
    intensity_scale_sparse = re.match(r"^.*_(\d*)outScaleSp",name)
    if intensity_scale_sparse is not None:
        intensity_scale_sparse = int(intensity_scale_sparse.groups()[0])
    else:
        intensity_scale_sparse = 1

    intensity_scale_dense = re.match(r"^.*_(\d*)outScaleD",name)
    if intensity_scale_dense is not None:
        intensity_scale_dense = int(intensity_scale_dense.groups()[0])
    else:
        intensity_scale_dense = 1
    return intensity_scale_dense,intensity_scale_sparse

def get_number_of_frames(name):
    n_frames = re.match(r"^.*_(\d*)timeF",name)
    if n_frames is not None:
        n_frames = int(n_frames.groups()[0])
    else:
        n_frames = 1
    return n_frames


def net_get_params(net):
    if hasattr(net, 'module'):
        return net.module
    else:
        return net


def volume_2_projections(vol_in, proj_type=torch.max, scaling_factors=[1,1,2], depths_in_ch=False, ths=[0.0,1.0], normalize=False, border_thickness=10, add_scale_bars=True, scale_bar_vox_sizes=[40,20]):
    vol = vol_in.detach().clone().abs()
    # Normalize sets limits from 0 to 1
    if normalize:
        vol -= vol.min()
        vol /= vol.max()
    if depths_in_ch:
        vol = vol.permute(0,2,3,1).unsqueeze(1)
    if ths[0]!=0.0 or ths[1]!=1.0:
        vol_min,vol_max = vol.min(),vol.max()
        vol[(vol-vol_min)<(vol_max-vol_min)*ths[0]] = 0
        vol[(vol-vol_min)>(vol_max-vol_min)*ths[1]] = vol_min + (vol_max-vol_min)*ths[1]

    vol_size = list(vol.shape)
    vol_size[2:] = [vol.shape[i+2] * scaling_factors[i] for i in range(len(scaling_factors))]

    if proj_type is torch.max or proj_type is torch.min:
        x_projection, _ = proj_type(vol.float().cpu(), dim=2)
        y_projection, _ = proj_type(vol.float().cpu(), dim=3)
        z_projection, _ = proj_type(vol.float().cpu(), dim=4)
    elif proj_type is torch.sum:
        x_projection = proj_type(vol.float().cpu(), dim=2)
        y_projection = proj_type(vol.float().cpu(), dim=3)
        z_projection = proj_type(vol.float().cpu(), dim=4)

    out_img = z_projection.min() * torch.ones(
        vol_size[0], vol_size[1], vol_size[2] + vol_size[4] + border_thickness, vol_size[3] + vol_size[4] + border_thickness
    )

    out_img[:, :, : vol_size[2], : vol_size[3]] = z_projection
    out_img[:, :, vol_size[2] + border_thickness :, : vol_size[3]] = F.interpolate(x_projection.permute(0, 1, 3, 2), size=[vol_size[-1],vol_size[-3]], mode='nearest')
    out_img[:, :, : vol_size[2], vol_size[3] + border_thickness :] = F.interpolate(y_projection, size=[vol_size[2],vol_size[4]], mode='nearest')


    if add_scale_bars:
        line_color = 1.0 #out_img.max()
        # Draw white lines
        out_img[:, :, vol_size[2]: vol_size[2]+ border_thickness, ...] = line_color
        out_img[:, :, :, vol_size[3]:vol_size[3]+border_thickness, ...] = line_color
        start = 0.02
        out_img[:, :, int(start* vol_size[2]):int(start* vol_size[2])+4, int(0.9* vol_size[3]):int(0.9* vol_size[3])+scale_bar_vox_sizes[0]] = line_color
        out_img[:, :, int(start* vol_size[2]):int(start* vol_size[2])+4, vol_size[2] + border_thickness + 10 : vol_size[2] + border_thickness + 10 + scale_bar_vox_sizes[1]*scaling_factors[2]] = line_color
        out_img[:, :, vol_size[2] + border_thickness + 10 : vol_size[2] + border_thickness + 10 + scale_bar_vox_sizes[1]*scaling_factors[2], int(start* vol_size[2]):int(start* vol_size[2])+4] = line_color

    return out_img

def imshow2D(img, blocking=False):
    plt.figure(figsize=(10,10))
    plt.imshow(img[0,0,...].float().detach().cpu().numpy())
    if blocking:
        plt.show()
def imshow3D(vol, blocking=False, normalize=True):
    plt.figure(figsize=(10,10))
    plt.imshow(volume_2_projections(vol.permute(0,2,3,1).unsqueeze(1), normalize=True)[0,0,...].float().detach().cpu().numpy())
    if blocking:
        plt.show()
def imshowComplex(vol, blocking=False):
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(volume_2_projections(torch.real(vol).permute(0,2,3,1).unsqueeze(1))[0,0,...].float().detach().cpu().numpy())
    plt.subplot(1,2,2)
    plt.imshow(volume_2_projections(torch.imag(vol).permute(0,2,3,1).unsqueeze(1))[0,0,...].float().detach().cpu().numpy())
    if blocking:
        plt.show()

def si(tensor, path='output.png'):
    return save_image(tensor, path)

def save_image(tensor, path='output.png'):
    if tensor.ndim<4:
        tensor = tensor.unsqueeze(0)
    if 'tif' in path:
        imsave(path, tensor[0,...].cpu().numpy().astype(np.float16))
        return
    if tensor.shape[1] == 1:
        imshow2D(tensor)
    else:
        imshow3D(tensor)
    plt.savefig(path)