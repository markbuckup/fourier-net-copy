# FOURIER-Net - Code Usage



### Training the Fourier-Net with any dataset

The main.py takes an argument of "dataset_path", where it expects the path to  the data folder. Any dataset that needs to be trained on must replicate the preprocessing done for the ACDC dataset. 

The directory structure of the dataset folder should be as follows

|Folder pointed to by the argument|
					|--------------radial_faster
								     |-------------- 256_resolution_10_spokes
														|-------------- patient_1-----------------------|-------vid_0.pth
														|							        |-------vid_1.pth
														|-------------- patient_2			     |.......
														|								|-------vid_10.pth
														|-------------- .......
														|
														|-------------- patient_150
														|
														|-------------- metadata.pth



The metadata.pth must be a .pth file stored by torch.save(.) and should contain a dictionary with the following keys:

- 'num_patients' = Number of Patients
- 'coil_masks' = A matrix of size (N_COILS_VARIANTS,N_COILS, RES, RES). Each variant is a set of Coils - gaussian masks with their centres scattered. Each variant is slightly rotated than the others
- 'coil_variant_per_patient_per_video' = We assign each patient and video a particular coil variant randomly - we store in the metadic to ensure consistency if we rerun the script
- 'GAs_per_patient_per_video' = We assign each frame of (patient and video) a set of acquisition angles - we store in the metadic to ensure consistency if we rerun the script
- 'num_vids_per_patient' = Number of slices for each patient - list of lists
- 'frames_per_vid_per_patient' = Number of frames / Cardiac phases for each patient (each slice must have the same number of cardiac phases)

The vid_i.pth for each patient should contain the following keys

- 'spoke_mask' = 0-1 mask containing ones at locations with newly acquired data
- 'targ_video' = The ground truth fully sampled and coil-combined video 
- 'coilwise_input' = The undersampled coilwise complex k-space input
- 'coilwise_targets' = The coil-wise ground truth - fully sampled



Post training, the fourier-net can be evaluated using the following commands:

``````bash
python3 main.py --run_id <run_id> --gpu <GPU IDs> --dataset_path <path_to_data>							# Start Training
python3 main.py --run_id <run_id> --gpu <GPU IDs> --dataset_path <path_to_data> --resume				# Resume Training
python3 main.py --run_id <run_id> --gpu <GPU IDs> --dataset_path <path_to_data> --resume --eval			# Evaluate Training - 																												Save predicted frames 																											  Print L1/L2/SSIM
python3 main.py --run_id <run_id> --gpu <GPU IDs> --dataset_path <path_to_data> \
				--resume --eval --visualise_only														# Save predicted frames

python3 main.py --run_id <run_id> --gpu <GPU IDs> --dataset_path <path_to_data> \
				--resume --eval --numbers_only															# Print L1/L2/SSIM
``````



### Generating a Video for Qualitative Analysis

For this, we need a full trained MDCNN experiment and a fully trained FOURIER-Net experiment.

We first save the raw images using 

``````bash
python3 main.py --run_id <mdcnn_id> --gpu <GPU ID> --resume --eval --visualise_only --raw_visual_only
python3 main.py --run_id <fouriernet_id> --gpu <GPU ID> --resume --eval --visualise_only --raw_visual_only
``````

After saving the raw images, you can create a video using the video_gen script

``````bash
python3 video_gen.py --mdcnn_id <mdcnn_id> --fouriernet_id <fouriernet_id>
``````



### Test the Fourier-Net with another dataset (Without Training)

This is for cases where we need to just test on a datset using some pretrained weights. You need a "data.pth" with a single key 'undersampled_data'. This data MUST be normalised so that the sum of squares of the k-space coils is in the range (0,1). 

This undersampled_data should have a torch complex matrix of shape (N, N_c, R, R ). Here N is the number of frames of undersampled data, N_c is the number of coils the FOURIER-Net checkpoints were trained with, and, R is the image resolution.

The usage of code is:

``````bash
python3 main.py --run_id <run_id of the fully trained experiment> 
				--gpu <GPU ID> 
				--resume 
				--eval_on_real 
				--actual_data_path <path to data.pth>
``````



Here, run_id is the fully trained experiment that we want to resume to evaluate the new dataset. 