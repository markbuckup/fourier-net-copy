import torch
import torchvision
import os
import random
import PIL
import argparse
import sys
sys.path.append('../../')
from tqdm import tqdm
import kornia
import pickle
from kornia.metrics import ssim
from torchvision import transforms
import matplotlib.pyplot as plt
import kornia
from kornia.utils import draw_line
import numpy as np
import nibabel as nib
from skimage.exposure import match_histograms

#######################################################################################################
#   AERS: Important! To preprocess data different than ACDC, edit line 107-112 and paths throughout   #
#######################################################################################################

import torchkbnufft as tkbn

from utils.functions import get_window_mask, get_coil_mask

# AERS: Sets the seed to 0 for numpy, pytorch, cuda, all GPUs. 
# Disables CuDNN (to avoid non-deterministic results). 
# Enables CuDNN to use deterministic algorithms. 
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch(0)

parser = argparse.ArgumentParser()
parser.add_argument('--resolution', type = int, default = 256)
parser.add_argument('--gpu', type = int, default = -1)
parser.add_argument('--pat_start', type = int, default = 1)
parser.add_argument('--pat_end', type = int, default = 150)
parser.add_argument('--save_path', type = str, default = '/Data/ExtraDrive1/niraj/datasets/ACDC')
parser.add_argument('--metadata_only', action = 'store_true')
args = parser.parse_args()

if args.gpu == -1:
    device = torch.device('cpu')
else:
    device = torch.device('cuda:{}'.format(args.gpu))

# AERS 20240716: Creates the targeted histogram. Randomly selected the first image of the first patient, in the 8th cardiac phase. 
# To-do: Test whole volume histogram or heart only?
# To-do: Normalize whole volume rather than each images in the volume?
histogram_target = nib.load('raw/patient001/patient001_4d.nii.gz').get_fdata()[:,:,0,7]

def read_nib_preprocess(path, heq = True):
    # AERS: Loads data   
    img = nib.load(path).get_fdata()

     # AERS: Initializes variables   
    drop_videos = []
    flag = False
    tot = 0

    # AERS: Identifies invalid frames: those that contain NaN or that are all zeros. 
    # Outputs: loaded images, a flag indicating if invalid frames were removed, and the number of frames that were removed.
    for vi in range(img.shape[2]):
        if (not np.isfinite(img[:,:,vi,:]).all()) or (img[:,:,vi,:].sum(0).sum(0) == 0).any():
            # drop video vi
            drop_videos.append(vi)
            flag = True

    # AERS: Removes invalid frames
    for di in reversed(sorted(drop_videos)):
        tot += 1
        img = np.concatenate((img[:,:,:di,:], img[:,:,di+1:,:]), 2)

    # AERS: Performs histogram equalization and normalizes each images from 0 to 1
    if heq:
        for i in range(img.shape[2]):
            for j in range(img.shape[3]):
                img[:,:,i,j] = match_histograms(img[:,:,i,j], histogram_target)
                img[:,:,i,j] = img[:,:,i,j] - img[:,:,i,j].min()
                img[:,:,i,j] = img[:,:,i,j] / img[:,:,i,j].max()
    return img, flag, tot

# AERS: When passes the patient index, it returns the patient path
def num_to_str(num):
    x = str(num)
    while len(x) < 3:
        x = '0' + x
    return './raw/patient{}/patient{}_4d.nii.gz'.format(x,x)

# AERS: Defines GAs
GR = (1 + (5**0.5))/2
GA = np.pi/GR

#######################################################################################################
# AERS: Important! Parameters for ACDC data. Needs to be changed for othe datasets.

NUM_PATIENTS = 150
RES=args.resolution
NUM_SPOKES = 10
N_COILS_VARIANTS = 100
N_COILS = 8
LOOP_FRAMES = 120
#######################################################################################################


# NRM: Define the Gridding and reverse gridding operations
kb_ob = tkbn.KbInterp(im_size=(512,512), grid_size = (1024,1024), numpoints = 6, kbwidth = 19.34, device = device)    # AERS: higher kbwidth is a smaller/sharper kernel
kbinterp = tkbn.KbInterpAdjoint(im_size=(256,256), grid_size = (256,256), numpoints = 6, kbwidth = 2.34, device = device) # AERS: In video, Niraj says that the number of points should be 6 for both lines, but it was originally 3.
recon_im_size = (256,256)
recon_grid_size = (1024,1024)
recon_num_points = 8
recon_kbwidth = 0.84
kbinterp2 = tkbn.KbInterpAdjoint(im_size=recon_im_size, grid_size = recon_grid_size, numpoints = recon_num_points, kbwidth = recon_kbwidth, device = device)
kbinterp_full = tkbn.KbInterpAdjoint(im_size=recon_im_size, grid_size = recon_grid_size, numpoints = 6, kbwidth = 9.34, device = device)


def get_omega2d(thetas, rad_max = 128):
    """
    Generates a 2D frequency grid for given angles (thetas) and a specified maximum radius (rad_max).

    Parameters:
    thetas (Tensor): A tensor of angles (in radians) for which the frequency grid will be generated.
    rad_max (int, optional): The maximum radius value for the frequency grid. Default is 128.

    Returns:
    Tensor: A tensor of shape (2, len(thetas), 2*rad_max+1) containing the frequency coordinates.
            The first dimension represents the x and y coordinates, the second dimension corresponds
            to the provided angles (thetas), and the third dimension contains the frequency values
            ranging from -pi to pi.

    The frequency grid (omega1) is generated by calculating the x and y coordinates based on the radii
    and angles, normalizing these coordinates by rad_max, and scaling by pi.

    Example:
    thetas = torch.tensor([0, np.pi/4, np.pi/2])
    omega1 = get_omega2d(thetas)
    """    
    omega1 = torch.zeros(2,thetas.shape[0],2*rad_max+1).to(device)

    for itheta, theta_rad in enumerate(thetas):

        rads = np.arange(-rad_max,rad_max+1)
        xs = rads*np.cos(theta_rad)
        ys = rads*np.sin(theta_rad)
        
        omega1[0,itheta,:] = torch.from_numpy(np.pi*(xs/rad_max)) # ?, in 0 axis omega, filled with xs, rad_max=256, every omega value will be between 0 and 1
        omega1[1,itheta,:] = torch.from_numpy(np.pi*(ys/rad_max)) # ?, in 1 axis omega, filled with ys
    return omega1


# NRM: Generate the omegas for a fully sampled image - will be used as ground truth
angles = GA*np.arange(1000)
omega_full = get_omega2d(angles, rad_max = 2048).to(device).reshape(2,-1)

# NRM: Generate the density compensation for this fully sampled kspace
dcomp_full = tkbn.calc_density_compensation_function(ktraj=omega_full, im_size=recon_im_size, grid_size = recon_grid_size, numpoints = 6, kbwidth = 9.34).to(device)

# NRM: Transformation to reduce resolution
# Any resolution less than 256 should be cropped to ensure the weights do not learn a shrinked representation
# AERS: For resolution <256, crop the image instead of resampling.
transform = torch.nn.Sequential(
    transforms.Resize((256,256), antialias = True),
    transforms.CenterCrop(RES),
)


# NRM: Write a metadictionary with the following attributes
# metadic['num_patients']                           = Number of Patients
# metadic['coil_masks']                             = A matrix of size (N_COILS_VARIANTS,N_COILS, RES, RES)
#                                                           Each variant is a set of Coils - gaussian masks with their centres scattered
#                                                           Each variant is slightly rotated than the others
# metadic['coil_variant_per_patient_per_video']     = We assign each patient and video a paritcular coil variant randomly - we store in the metadic to ensure consistency if we rerun the script
# metadic['GAs_per_patient_per_video']              = We assign each frame of (patient and video) a set of acquisition angles - we store in the metadic to ensure consistency if we rerun the script
# metadic['num_vids_per_patient']                   = Number of slices for each patient - list of lists
# metadic['frames_per_vid_per_patient']             = Number of frames / Cardiac phases for each patient (each slice must have the same number of cardiac phases)

# AERS: Metadata will be different for different parameters, so it will create separate folder if these are changed
metapath = 'radial_faster/{}_resolution_{}_spokes/metadata.pth'.format(RES, NUM_SPOKES)

# AERS: If metadata exists, load it
if os.path.isfile(metapath):
    metadic = torch.load(metapath)
    print("Meta Data Loaded")

# AERS: If metadata doesn't exists, create it
else:
    # AERS: Saves metadata
    os.makedirs(os.path.join(args.save_path, 'radial_faster/{}_resolution_{}_spokes'.format(RES, NUM_SPOKES)), exist_ok = True)
    metadic = {}
    metadic['num_patients'] = NUM_PATIENTS
    # AERS: Number of coil variantes is the predefined, equally rotated, number of coils. The initial angle is randomly selected. I think the N_COILS are equally distributed around 2π.
    metadic['coil_masks'] = torch.zeros((N_COILS_VARIANTS,N_COILS, RES, RES)) # AERS: 100, 8, 128, 128. 
    for i in range(N_COILS_VARIANTS):
        #AERS: Here is where theta_init (initial coil location) gets generated for each patient.
        metadic['coil_masks'][i] = get_coil_mask(theta_init = (np.pi/4)*i*(1/N_COILS_VARIANTS), n_coils = N_COILS, resolution = RES) 

    # AERS: Saves i so that the same coil variant (theta_init) can ge generated if needed. This way, the coils are always the same for a speciic patient.
    metadic['coil_variant_per_patient_per_video'] = [[] for i in range(NUM_PATIENTS)]

    # AERS: Saves the number of GAs per dataset, number of videos (varies in ACDC dataset), and frames.
    metadic['GAs_per_patient_per_video'] = [[] for i in range(NUM_PATIENTS)]
    metadic['num_vids_per_patient'] = [0 for i in range(NUM_PATIENTS)]
    metadic['frames_per_vid_per_patient'] = [0 for i in range(NUM_PATIENTS)]
    
    #AERS: Loop over patients
    for p_num in tqdm(range(NUM_PATIENTS), desc = 'Writing Metadata patient data'):
        path = num_to_str(p_num+1) # AERS: Returns the path to patient folder based on index (patient number)

        # Loads original data and performs histogram equalization
        d, flag, dvid = read_nib_preprocess(path)
        if flag:
            print("Patient {} flagged, {} video(s) dropped".format(i, dvid))
        d = torch.permute(torch.FloatTensor(d).unsqueeze(4), (2,3,4,0,1))
        num_vids, num_frames, n_coils, n_row, n_col = d.shape

        metadic['num_vids_per_patient'][p_num] = num_vids
        metadic['frames_per_vid_per_patient'][p_num] = num_frames

        for vi in range(num_vids):
            ga_random_start = np.random.randint(377) # AERS: The GA 0 and 377 are very similar. So this is used kind of like a loop (per Niraj).

            # AERS: For a single slice, if it has 30 frames and you want to make 120 to simulate longer acquisitions, each frame has a particular set of GAs.
            # This code stores those GAs in GAs_per_patient_per_video.
            # GAs_per_patient_per_video is a list of lists, with a random start between 0 and 377.
            # This is also saved to replicate the random starting point when needed.

            metadic['GAs_per_patient_per_video'][p_num].append((ga_random_start+np.arange(LOOP_FRAMES*NUM_SPOKES).reshape(LOOP_FRAMES,NUM_SPOKES))%377)
            if not metadic['coil_variant_per_patient_per_video'][p_num]:
                metadic['coil_variant_per_patient_per_video'][p_num] = [0 for i in range(num_vids)]
            metadic['coil_variant_per_patient_per_video'][p_num][vi] = np.random.randint(N_COILS_VARIANTS)

    # AERS: Saves metadata
    torch.save(metadic, metapath) # AERS: As of 20240717, metadata is always w in radial_faster/###_resolution_##_spokes
    print("Meta Data Generated")

# AERS: Main FOR loop. Loads the metadata
if not args.metadata_only: 
    for p_num in range(args.pat_start-1, args.pat_end): # AERS: For ACDC, 1 to 150 (indexes the patient folder name)
        # NRM: Create the folder to save the processed video
        # AERS: One folder for each patient. With n number of slices, saved as .pth files (complex data). Assumed to be the coil combine image.
        folder_name = os.path.join(args.save_path, 'radial_faster/{}_resolution_{}_spokes/patient_{}'.format(RES, NUM_SPOKES,p_num+1))
        os.makedirs(folder_name, exist_ok = True)

        path = num_to_str(p_num+1)
        # NRM: Read in the video
        d_inp, flag, dvid = read_nib_preprocess(path)
        
        # AERS: Prints if videos/frames were dropped. This happens when they are invalid: 1) contains NaN, or 2) they are all 0s
        if flag:
            print("Patient {} flagged, {} video(s) dropped".format(p_num, dvid))
        d_inp = torch.permute(torch.FloatTensor(d_inp).unsqueeze(4), (2,3,4,0,1))
        num_vids, num_frames, n_coils, n_row, n_col = d_inp.shape # NRM: Reshape

        # NRM: Pad the image with zeros to create a square image
        d_padded = torch.zeros(num_vids, num_frames, n_coils, max(n_row, n_col),max(n_row, n_col))
        row_pad = (max(n_row, n_col) - n_row)//2
        col_pad = (max(n_row, n_col) - n_col)//2
        d_padded[:,:,:,row_pad:row_pad + n_row, col_pad:col_pad+n_col] = d_inp
        del d_inp

        # NRM: Center crop the image according to the required resoltuion and scale it 0-1
        # AERS: This is performed on the padded square image, not the original (d_inp) which was deleted above
        d = transform(d_padded.reshape(num_vids*num_frames, 1, max(n_row,n_col), max(n_row,n_col))).reshape(num_vids, num_frames, n_coils, RES, RES)
        d = d - d.min(-1)[0].min(-1)[0].unsqueeze(-1).unsqueeze(-1)
        d = d / (1e-10 + d.max(-1)[0].max(-1)[0].unsqueeze(-1).unsqueeze(-1))
        # AERS:To-do? If data were coming from scanner, they are not coil combined. Per Niraj, the images before coil combination, should sum up to 1. 

        # AERS: Number of videos is the number of slices. A dictionary is saved for each video/slice.
        for vi in tqdm(range(num_vids), desc = 'Running for Patient {}'.format(p_num+1)):
            dic = {}
            dic['targ_video'] = d[vi,np.arange(LOOP_FRAMES)%num_frames] # NRM: FRAMES, 1, RES, RES  <-------------------------------- AERS: Ground truth
            dic['spoke_mask'] = torch.zeros((LOOP_FRAMES, 1, RES, RES)).type(torch.uint8) # NRM: FRAMES, COILS, RES, RES <----------- AERS: Location where new data are acquired
            # dic['coilwise_input_mag'] = torch.zeros((LOOP_FRAMES, N_COILS, RES, RES)).type(torch.uint8) # FRAMES, COILS, RES, RES
            dic['coilwise_input'] = torch.zeros((LOOP_FRAMES, N_COILS, RES, RES)) # NRM: FRAMES, COILS, RES, RES
            dic['coilwise_input'] = torch.complex(dic['coilwise_input'],dic['coilwise_input'])
            dic['coilwise_target'] = torch.zeros((LOOP_FRAMES, N_COILS, RES, RES)).type(torch.uint8) # NRM:FRAMES, COILS, RES, RES

            # AERS: Iterate over the number of frames
            for fi in range(LOOP_FRAMES):

                # NRM: Fetch the Coil Variant from the metadictionary
                coil_variant = metadic['coil_variant_per_patient_per_video'][p_num][vi]
                curr_coil = metadic['coil_masks'][coil_variant] # AERS: 8x256x256

                # NRM: Get the coilwise ground truth
                input_frame = dic['targ_video'][fi,:,:,:] * curr_coil # AERS: input_frame is 1x256x256, after product it will be 8x256x256

                # NRM: Important! The kspace data should be such that the sum of squares of the input is in the range (0,1)
                temp = ((input_frame**2).sum(0)**0.5)
                input_frame = input_frame / (1e-10 + temp.max())

                # NRM: Fetch the sampled angles from the metadictionary
                angle_indices = metadic['GAs_per_patient_per_video'][p_num][vi][fi]
                angles = GA*angle_indices

                # NRM: Generate the omegas - x,y coordinates 
                omega1 = get_omega2d(angles, rad_max = 2048).to(device).reshape(2,-1)

                # NRM: Pad the image with zeros to prevent artefacts and the the fft
                padded_frame = torch.zeros(1, N_COILS, 2*input_frame.shape[1],2*input_frame.shape[2])
                padded_frame[0,:,(RES//2):(RES//2)+RES,(RES//2):(RES//2)+RES] = input_frame
                padded_frame = torch.fft.fft2(padded_frame)
                
                # NRM: Reverse Gridding operation - obtain data at non grid coordiates
                y = kb_ob(padded_frame.to(device),omega1) # AERS: y contains the k-space values for each pair of omega1 (x,y real coordinates), so it is complex

                # NRM: Do a similar operation for a fully sampled image - 1000 spokes
                y_full = kb_ob(padded_frame.to(device),omega_full)

                # NRM: Compute the density compensation weights for the undersampled spokes
                dcomp_under = tkbn.calc_density_compensation_function(ktraj=omega1, im_size=recon_im_size, grid_size = recon_grid_size, numpoints = recon_num_points, kbwidth = recon_kbwidth).to(device)
                
                # NRM: If we grid the density compensation weights and threshold them, we get the mask of the spokes
                # AERS: this is doing the same as the gridding below, but without k-space data. kbinterp takes the density compensation, 
                # which is >0 near spokes and then it gets thresholded to generate the spoke mask. 
                dic['spoke_mask'][fi,0,:,:] = torch.fft.fftshift(kbinterp(dcomp_under, omega1)[0,:].abs().squeeze(), dim = (-2,-1)).cpu() > 0.1 

                # NRM: Gridding operation for both the undersampled and fully sampled image
                myfft_interp = torch.fft.fftshift(kbinterp2(dcomp_under*y, omega1)[0,:].cpu(), dim = (-2,-1)) # AERS: kbinterp2 returns a gridded version of y (k-space), inverse NUFFT
                myfft_interp_full = torch.fft.fftshift(kbinterp_full(dcomp_full*y_full, omega_full)[0,:].cpu(), dim = (-2,-1))
                
                # NRM: Crop the image
                myfft_interp = myfft_interp[:,::4,::4]
                myfft_interp_full = myfft_interp_full[:,::4,::4]

                # NRM: Scale both the undersampled and fully sampled inputs so that the sum of squares of the ifft are in the range (0,1)                
                temp = torch.fft.ifft2(torch.fft.ifftshift(myfft_interp_full, dim = (-2,-1)))
                temp = temp - temp.abs().min(2)[0].min(1)[0].unsqueeze(-1).unsqueeze(-1)
                scaling_factor = 1/(1e-10 + ((temp.abs()**2).sum(0)**0.5).max())
                temp = temp * scaling_factor

                # NRM: temp now has k-space data such that the sum of squares of the ifft are in the range(0,1)
                temp = torch.fft.fftshift(torch.fft.fft2(temp), dim = (-2,-1))
                
                # NRM: Store the DC component and temporarily set it to 0
                true_dc = temp[:,RES//2,RES//2].clone()
                temp[:,RES//2,RES//2] = 0
                myfft_interp_full[:,RES//2,RES//2] = 0
                myfft_interp[:,RES//2,RES//2] = 0

                # NRM: Normalise the interpolated FFTs such that the mean of the 3x3 region near their center matches temp (computed above)
                myfft_interp_full = myfft_interp_full * (temp.abs()[:,127:130,127:130].mean(-2, keepdim = True).mean(-1, keepdim = True) / (myfft_interp_full.abs()[:,127:130,127:130].mean(-2, keepdim = True).mean(-1, keepdim = True) + 1e-10))
                myfft_interp = myfft_interp * (temp.abs()[:,127:130,127:130].mean(-2, keepdim = True).mean(-1, keepdim = True) / (myfft_interp.abs()[:,127:130,127:130].mean(-2, keepdim = True).mean(-1, keepdim = True) + 1e-10))
                # NRM: Restore the DC values according to temp (computed above)
                myfft_interp_full[:,RES//2,RES//2] = true_dc
                myfft_interp[:,RES//2,RES//2] = true_dc

                # NRM: Store the complex coilwise undersampled input  
                dic['coilwise_input'][fi] = myfft_interp
                # NRM: Store coilwise targets - storing as uint8 to conserve memory
                dic['coilwise_target'][fi] = (torch.fft.ifft2(torch.fft.ifftshift(myfft_interp_full, dim = (-2,-1))).abs().clip(0,1)*255).type(torch.uint8)

            # NRM: Store the target (ground truth) - storing as uint8 to conserve memory
            dic['targ_video'] = (dic['targ_video']*255).type(torch.uint8)
            torch.save(dic, os.path.join(folder_name, 'vid_{}.pth'.format(vi)))
