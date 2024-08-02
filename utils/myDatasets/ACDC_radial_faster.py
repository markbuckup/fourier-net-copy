import os
import torch
import random
import numpy as np
import torchvision
import matplotlib.pyplot as plt

from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset

EPS = 1e-10
CEPS = torch.complex(torch.tensor(EPS),torch.tensor(EPS)).exp()

def seed_torch(seed=0):
    """
    Sets the random seed for reproducibility across various libraries.

    Parameters:
    -------------
    - seed : int, optional
        The seed value to use. 
            Defaults to 0.

    ================================================================================================
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class ACDC_radial_ispace(Dataset):
    """
    Custom dataset for ACDC radial data with in-memory storage optimization.
    A memoised version of the ACDC_radial class defined further is this module

    Attributes:
    -------------
    - mem_inputs : list
        List of tensors to store inputs.
    - mem_inputs_min : list
        List of tensors to store minimum values of inputs.
    - mem_inputs_max : list
        List of tensors to store maximum values of inputs.
    - mem_gts : list
        List of tensors to store ground truth data.
    - mem_stored_bool : list
        List of boolean flags to indicate stored data.
    - data_init_done : bool
        Flag to indicate if data initialization is done.
    
    Methods:
    -------------
    - set_data(input, gt, pat_id, vid_id)x: Sets the data for a given patient and video ID.
    - get_data(pat_id, vid_id): Retrieves the data for a given patient and video ID.
    - check_data(pat_id, vid_id): Checks if data for a given patient and video ID is stored.
    - bulk_set_data(actual_pnums, vnums, inputs, gts): Sets data for multiple patients and videos in bulk.
    - data_init(whole_num_vids_per_patient, parameters): Initializes data storage.
    - __init__(path, parameters, device, train=True, visualise_only=False): Initializes the dataset.
    - __getitem__(i): Retrieves an item from the dataset.
    - __len__(): Returns the length of the dataset.

    ==============================================================================================================
    """

    # AERS: These class variables are created outside the functions, and each instance of the ACDC_radial_ispace class will have access to these variable (basically like global variables). 
    # This way, all dataloader workers (n=8/GPU in params.py) can access the memoized data without creating individual datasets (so memory doesn't run out.)
    # These variables are initialized in data_init class method

    mem_inputs = []
    mem_inputs_min = []
    mem_inputs_max = []
    mem_gts = []
    mem_stored_bool = []
    data_init_done = False   # AERS: To avoid loading the data more than once, data_init_done is set to False. Then, once it is initialized and data are loaded, the data_init_done is set to True. Won't be repeated.
    
    @classmethod
    def set_data(cls, input, gt, pat_id, vid_id):
        """
        AERS:
        Sets and scales input data, then stores it along with the ground truth data.

        This class method takes input data and ground truth data, scales the input, and stores both the scaled 
        input and the ground truth data as 8-bit unsigned integers. The data is stored in memory, indexed by 
        patient ID (`pat_id`) and video ID (`vid_id`).

        Parameters:
        ------------
        - input : torch.Tensor
            The input data tensor to be scaled and stored. The tensor is expected to be multi-dimensional 
            (e.g., a video or image stack).
        - gt : torch.Tensor
            The ground truth data tensor to be stored, typically in the same shape as `input`.
        - pat_id : int or str
            The patient ID used to index the data in memory.
        - vid_id : int or str
            The video ID used to index the data in memory.

        Notes:
        -------
        - The input data is scaled to the range [0, 255] after subtracting the minimum value and dividing by the maximum value to ensure normalized intensity. The scaled data is then stored as 8-bit unsigned integers.
        - The ground truth data is also scaled to the range [0, 255] and stored as 8-bit unsigned integers.
        - This method updates the class-level memory (`cls.mem_inputs`, `cls.mem_gts`, etc.) for the corresponding patient and video IDs.

        ====================================================================================================================
        """
        # print('setting', pat_id, vid_id)
        
        cls.mem_inputs_max[pat_id][vid_id] = input.max(-1)[0].max(-1)[0]
        cls.mem_inputs_min[pat_id][vid_id] = input.min(-1)[0].min(-1)[0]

        input_scaled = input - input.min(-1, keepdim = True)[0].min(-2, keepdim = True)[0]
        input_scaled = input / (1e-10 + input.max(-1, keepdim = True)[0].max(-2, keepdim = True)[0])

        cls.mem_inputs[pat_id][vid_id] = (input_scaled*255).type(torch.uint8)
        cls.mem_gts[pat_id][vid_id] = (gt*255).type(torch.uint8)
        cls.mem_stored_bool[pat_id][vid_id] = 1

    @classmethod
    def get_data(cls, pat_id, vid_id):
        """
        AERS: 
        Retrieves and reconstructs the stored input and ground truth data.

        This class method retrieves the scaled input and ground truth data for a specific patient ID (`pat_id`) 
        and video ID (`vid_id`) from the class-level memory. The method then reconstructs the original input 
        data by applying the inverse of the scaling operation.

        Parameters:
        ------------
        - pat_id : int or str
            The patient ID used to retrieve the data from memory.
        - vid_id : int or str
            The video ID used to retrieve the data from memory.

        Returns:
        --------
        - tuple of torch.Tensor
            A tuple containing:
            - `ret_inp` (torch.Tensor): The reconstructed input data tensor.
            - `ret_gt` (torch.Tensor): The ground truth data tensor.
        
        Raises:
        -------
        - AssertionError
            If the data for the specified `pat_id` and `vid_id` has not been stored yet.
        
        Notes:
        ------
        - The input data is reconstructed from its stored scaled version by reversing the normalization process.
        - The ground truth data is also retrieved and returned as a float tensor.

        ====================================================================================================================
        """
        # print('loading', pat_id, vid_id)

        maxs = (cls.mem_inputs_max[pat_id][vid_id]).unsqueeze(-1).unsqueeze(-1)
        mins = (cls.mem_inputs_min[pat_id][vid_id]).unsqueeze(-1).unsqueeze(-1)

        assert(cls.mem_stored_bool[pat_id][vid_id] == 1)
        ret_inp = (cls.mem_inputs[pat_id][vid_id]).float()/255. 
        ret_inp = ret_inp * (maxs-mins)
        ret_inp = ret_inp + mins
        ret_gt = (cls.mem_gts[pat_id][vid_id]).float()/255.
        return ret_inp, ret_gt

    @classmethod
    def check_data(cls, pat_id, vid_id):          # AERS: checks if these data are stored yet?
        """
        AERS:
        Checks if the data for a specific patient and video ID is stored in memory.

        This class method verifies whether the input and ground truth data for the given patient ID (`pat_id`) 
        and video ID (`vid_id`) have already been stored in the class-level memory.

        Parameters:
        -----------
        - pat_id : int or str
            The patient ID used to check the data in memory.
        - vid_id : int or str
            The video ID used to check the data in memory.

        Returns:
        --------
        - bool
            Returns `True` if the data for the specified `pat_id` and `vid_id` is stored, otherwise `False`.
        
        Notes:
        ------
        - This method is useful for determining whether data retrieval or initialization processes should be triggered.

        ====================================================================================================================
        """
        # print('checking', pat_id, vid_id)
        return cls.mem_stored_bool[pat_id][vid_id] == 1

    @classmethod
    def bulk_set_data(cls, actual_pnums, vnums, inputs, gts):
        """
        AERS:
        Sets and stores input and ground truth data in bulk for multiple patients and videos.

        This class method iterates over a batch of input and ground truth data, storing each item individually for the corresponding patient ID (`pnum`) and video ID (`vnum`).

        Parameters:
        -----------
        - actual_pnums : list of int or str
            A list of patient IDs corresponding to the batch of data.
        - vnums : list of int or str
            A list of video IDs corresponding to the batch of data.
        - inputs : torch.Tensor
            A batch of input data tensors with shape `(B, 120, Chan, 256, 256)`, where `B` is the batch size.
        - gts : torch.Tensor
            A batch of ground truth data tensors with shape `(B, 120, 1, 256, 256)`, where `B` is the batch size.

        Raises:
        -------
        - AssertionError
            If the `inputs` or `gts` tensors do not have 5 dimensions as expected.

        Notes:
        ------
        - This method ensures that the data for each patient and video in the batch is individually scaled and stored using the `set_data` method.
        - The 120 frames were set by NRM's experiment for ACDC dataset.

        ====================================================================================================================
        """
        assert(len(inputs.shape) == 5)
        assert(len(gts.shape) == 5)
        for i,(pnum, vnum) in enumerate(zip(actual_pnums, vnums)):
            # input shape = B, 120, Chan, 256, 256
            # GT shape = B, 120, 1, 256, 256
            cls.set_data(inputs.cpu()[i], gts.cpu()[i], pnum, vnum)

    @classmethod            # AERS: class methods is used here because for image space data, each GPU will have a separate copy
    def data_init(cls, whole_num_vids_per_patient, parameters):
        """
        AERS:
        Initializes the data storage for each patient and video.

        This class method sets up the data structures needed to store input and ground truth data 
        for each patient and video during training or testing. The method initializes the memory 
        with zeros or empty structures based on the provided parameters.

        Parameters:
        -----------
        - whole_num_vids_per_patient : list of int
            A list where each entry represents the number of videos per patient.
        - parameters : dict
                A dictionary of parameters that control the initialization process. 
                Expected keys include:
                
                - 'memoise_ispace' (bool): Flag indicating whether to memoize image space data.
                - 'coil_combine' (str): Method for coil combination ('SOS' or other methods).
                - 'init_skip_frames' (int): Number of initial frames to skip.
                - 'max_frames' (int): Maximum number of frames per video.
                - 'num_coils' (int): Number of coils for imaging.
                - 'image_resolution' (int): The resolution of the image (width and height in pixels).

        Notes:
        ------
        - This method uses class-level memory (e.g., `cls.mem_inputs`, `cls.mem_gts`) to store data structures for each patient and video.
        - The method also uses shared memory to allow multiple processes to access the data concurrently.
        - This method only runs once to avoid re-initializing the data (controlled by `cls.data_init_done`).

        ====================================================================================================================
        """
        if cls.data_init_done:
            return
        for n_vids in whole_num_vids_per_patient:  # AERS: Load all data for training or testing
            cls.mem_inputs.append([])              # AERS: Data initialization with zeros below. This needs to be modified if we want the image output to remain complex (not just the mag of complex data).
            cls.mem_inputs_max.append([])
            cls.mem_inputs_min.append([])
            cls.mem_gts.append([])
            cls.mem_stored_bool.append([])
            for vi in range(n_vids):
                if parameters['memoise_ispace']:
                    if parameters['coil_combine'] == 'SOS':
                        cls.mem_inputs[-1].append(torch.zeros(cls.max_frames - parameters['init_skip_frames'],1,parameters['image_resolution'],parameters['image_resolution']).type(torch.uint8))
                        cls.mem_inputs_max[-1].append(torch.zeros(cls.max_frames - parameters['init_skip_frames'],1))
                        cls.mem_inputs_min[-1].append(torch.zeros(cls.max_frames - parameters['init_skip_frames'],1))
                    else:
                        cls.mem_inputs[-1].append(torch.zeros(cls.max_frames - parameters['init_skip_frames'],parameters['num_coils'],parameters['image_resolution'],parameters['image_resolution']).type(torch.uint8))
                        cls.mem_inputs_max[-1].append(torch.zeros(cls.max_frames - parameters['init_skip_frames'],parameters['num_coils']))
                        cls.mem_inputs_min[-1].append(torch.zeros(cls.max_frames - parameters['init_skip_frames'],parameters['num_coils']))
                    cls.mem_gts[-1].append(torch.zeros(cls.max_frames - parameters['init_skip_frames'],1,parameters['image_resolution'],parameters['image_resolution']))
                cls.mem_stored_bool[-1].append(torch.zeros(1))

            [x.share_memory_() for x in cls.mem_inputs[-1]]         # AERS: Shares data to all workers
            [x.share_memory_() for x in cls.mem_gts[-1]]
            [x.share_memory_() for x in cls.mem_stored_bool[-1]]

        cls.data_init_done = True

    def __init__(self, path, parameters, device, train = True, visualise_only = False):
        """
        Initializes the ACDC_radial_ispace dataset.

        Parameters:
        -------------
        - path : str
            Path to the dataset.
        - parameters : dict
            Dictionary of parameters for dataset configuration.
        - device : torch.device
            Device to use for computations.
        - train : bool, optional
            Whether the dataset is for training. 
                Defaults to True.
        - visualise_only : bool, optional
            Whether to only visualize the data. 
                Defaults to False.

        ==========================================================================
        """

        super(ACDC_radial_ispace, self).__init__()
        ACDC_radial_ispace.max_frames = 120
        self.train = train
        self.parameters = parameters.copy()
        self.parameters['loop_videos'] = ACDC_radial_ispace.max_frames
        self.parameters['init_skip_frames'] = 90
        # AERS: Creates ACDC_radial as a subvariable used to speed up training by not generating multi-coil output again and again. This class stores those multi-coil outputs. 
        self.orig_dataset = ACDC_radial(path, self.parameters, device, train = train, visualise_only = visualise_only)        
        self.total_unskipped_frames = self.parameters['num_coils']*((ACDC_radial_ispace.max_frames - self.parameters['init_skip_frames'])*self.orig_dataset.num_vids_per_patient).sum()


    def __getitem__(self, i): # AERS: Only image space data are memoized
        """
        Retrieves an item from the dataset.

        Parameters:
        -------------
        - i : int
            Index of the item to retrieve.

        Returns:
        -----------
        - tuple
            A tuple containing the memory flag, data, and ground truth.

        ==========================================================================
        """
        if not ACDC_radial_ispace.data_init_done:                  # AERS: First thing it does is to make sure that the initialization is done
            ACDC_radial_ispace.data_init(self.orig_dataset.whole_num_vids_per_patient, self.parameters)
        pat_id, vid_id = self.orig_dataset.index_to_location(i)
        pat_id += self.orig_dataset.offset
        if self.train:
            assert(pat_id < 120)
        else:
            assert(pat_id >= 120)

        if ACDC_radial_ispace.check_data(pat_id, vid_id):          # AERS: Checks if data are stored in memory
            mem = torch.tensor(1)
            data, gt = ACDC_radial_ispace.get_data(pat_id, vid_id) # AERS: If yes, get those data
            return mem, data, gt
        else:
            mem = torch.tensor(0)                                  # AERS: If not, it will return mem and the original input from the original dataset
            return mem, *self.orig_dataset[i]                      # AERS: If mem = 0, expects huge dataset. If mem = 1, it expects data and gt. Calls the get item of the ACDC_radial class.
                                                                   # AERS: set_data is called from the training script, not here (actually uses bulk_set_data).

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
        int: The length of the dataset.
        """
        return len(self.orig_dataset)

class ACDC_radial(Dataset):
    """
    Custom dataset for ACDC radial data.

    Methods:
    ----------
    - index_to_location(i): Converts a flat index to patient and video IDs.
    - __init__(path, parameters, device, train=True, visualise_only=False): Initializes the dataset.
    - __getitem__(i): Retrieves an item from the dataset.
    - __len__(): Returns the length of the dataset.
    
    ===================================================================================================
    """
    def __init__(self, path, parameters, device, train = True, visualise_only = False): # AERS: Each video has 150 frames, 120 are used in training and 30 are used in testing. 
        # AERS: ACTUALLY, as per Niraj, June 7th 11:30AM video 53:07. He doesn't use 120 frames for training because the memory runs out, so he used 32 defined in params.py 'loop_videos'
    
        """
        Initializes the ACDC_radial dataset.

        Parameters:
        -------------
        - path : str
            Path to the dataset.
        - parameters : dict
            Dictionary of parameters for dataset configuration.
        - device : torch.device
            Device to use for computations.
        - train : bool, optional
            Whether the dataset is for training. 
                Defaults to True.
        - visualise_only : bool, optional
            Whether to only visualize the data. 
                Defaults to False.

        ====================================================================================================
        """
        super(ACDC_radial, self).__init__()
        self.path = path
        self.train = train
        self.parameters = parameters
        self.device = device
        self.train_split = parameters['train_test_split']
        
        self.final_resolution = parameters['image_resolution']
        self.resolution = 256
        
        self.num_coils = parameters['num_coils']
        assert(self.num_coils == 8)

        self.ft_num_radial_views = parameters['kspace_num_spokes']

        if visualise_only:
            self.loop_videos = 120
        else:
            self.loop_videos = parameters['loop_videos']
        
        self.data_path = os.path.join(self.path, 'radial_faster/{}_resolution_{}_spokes'.format(self.final_resolution, self.ft_num_radial_views))

        # AERS: Loads metadata and all relevant info to ACDC data
        metadic = torch.load(os.path.join(self.data_path, 'metadata.pth'))
        
        self.num_patients = metadic['num_patients']
        self.num_vids_per_patient = np.array(metadic['num_vids_per_patient'])
        self.coil_masks = metadic['coil_masks']
        self.coil_variant_per_patient_per_video = metadic['coil_variant_per_patient_per_video']
        self.GAs_per_patient_per_video = metadic['GAs_per_patient_per_video']

        self.whole_num_patients = self.num_patients
        self.whole_num_vids_per_patient = self.num_vids_per_patient.copy()

        self.actual_frames_per_vid_per_patient = np.array(metadic['frames_per_vid_per_patient'])
        self.frames_per_vid_per_patient = np.array(metadic['frames_per_vid_per_patient'])
        
        if self.train:
            self.offset = 0
            self.num_patients = int(self.train_split*self.num_patients)
        else:
            self.offset = int(self.train_split*self.num_patients)
            self.num_patients = self.num_patients - int(self.train_split*self.num_patients)
        
        if self.parameters['acdc_debug_mini']:
            self.num_vids_per_patient *= 0
            self.num_vids_per_patient += 1

        self.num_vids_per_patient = self.num_vids_per_patient[self.offset:self.offset+self.num_patients]
        self.frames_per_vid_per_patient = self.frames_per_vid_per_patient[self.offset:self.offset+self.num_patients]
        self.actual_frames_per_vid_per_patient = self.actual_frames_per_vid_per_patient[self.offset:self.offset+self.num_patients]
        self.GAs_per_patient_per_video = self.GAs_per_patient_per_video[self.offset:self.offset+self.num_patients]
        self.coil_variant_per_patient_per_video = self.coil_variant_per_patient_per_video[self.offset:self.offset+self.num_patients]

        if self.loop_videos != -1:
            self.frames_per_vid_per_patient *= 0
            self.frames_per_vid_per_patient += self.loop_videos
        self.vid_frame_cumsum = np.cumsum(self.num_vids_per_patient*self.frames_per_vid_per_patient)
        self.vid_cumsum = np.cumsum(self.num_vids_per_patient)

        self.num_videos = int(sum(self.num_vids_per_patient))

        self.total_unskipped_frames = self.parameters['num_coils']*((self.frames_per_vid_per_patient-self.parameters['init_skip_frames'])*self.num_vids_per_patient).sum()


    def index_to_location(self, i): # AERS: See comment in getitem. i returns the patient number and video number
        """
        Converts a flat index to patient and video IDs.

        Parameters:
        --------------
        - i : int
            The flat index.

        Returns:
        ----------
        - tuple
            The patient and video IDs corresponding to the flat index.
            
        ====================================================================================================
        """
        p_num = (self.vid_cumsum <= i).sum()
        if p_num == 0:
            v_num = i
        else:
            v_num = i - self.vid_cumsum[p_num-1]

        return p_num, v_num

    def __getitem__(self, i): # AERS: i indexes the video number for each patient. There is a cummulative sum of the number of videos for each patient (index_to_location). 
        """
        Retrieves an item from the dataset.

        Parameters:
        -------------
        - i : int
            The index of the item to retrieve.

        Returns:
        -----------
        - tuple
            A tuple containing various data tensors related to the item.

        ====================================================================================================
        """

        p_num, v_num = self.index_to_location(i)
        actual_pnum = self.offset + p_num  # There is an offset to index the video frames. If we are training the network, the offset is 0. If we are testing the network, the offset is 120. 
        index_coils_used = self.coil_variant_per_patient_per_video[p_num][v_num]
        coils_used = torch.flip(self.coil_masks[index_coils_used].unsqueeze(0), dims = (-2,-1))

        GAs_used = self.GAs_per_patient_per_video[p_num][v_num][:self.loop_videos,:]

        dic_path = os.path.join(self.data_path, 'patient_{}/vid_{}.pth'.format(actual_pnum+1, v_num))  # AERS: Sets the path where data are stored
        dic = torch.load(dic_path, map_location = torch.device('cpu'))                                 # AERS: Loads dictionary data, masks, ground truth and coilwise inputs, orignal number of frames (period)
        masks_applicable = dic['spoke_mask'].type(torch.float32)[:self.loop_videos,:,:,:]              # AERS: Note that this is loading the first 'loop_videos' as defined in params.py
        og_video = ((dic['targ_video']/255.)[:self.loop_videos,:,:,:])
        undersampled_fts = dic['coilwise_input'][:self.loop_videos,:,:,:]

        if 'coilwise_targets' not in dic:
            coilwise_targets = og_video * coils_used
            temp = ((coilwise_targets**2).sum(2)**0.5).max(-1)[0].max(-1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            coilwise_targets = coilwise_targets / (1e-10 + temp)
        else:
            coilwise_targets = (dic['coilwise_targets']/255.)[:self.loop_videos,:,:,:]

        Nf = self.actual_frames_per_vid_per_patient[p_num]

        return torch.tensor([actual_pnum, v_num]), masks_applicable, og_video, coilwise_targets, undersampled_fts, coils_used, Nf

        
    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
        -----------
        - int 
            The length of the dataset.

        """
        return self.num_videos 