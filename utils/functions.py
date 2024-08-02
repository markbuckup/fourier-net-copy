import torch
import torchvision
import os
import PIL
import sys
sys.path.append('../')
import numpy as np
from tqdm import tqdm
import kornia
from kornia.metrics import ssim
from torchvision import transforms
import matplotlib.pyplot as plt
import kornia
from kornia.utils import draw_line
from torch import nn, optim
import utils

EPS = 1e-10
GR = (1 + (5**0.5))/2
GA = np.pi/GR

# AERS: Generates GA masks for a finite number of GAs (n=376), because GA of n=0 and n=377 are very similar.
def get_golden_bars(num_bars = 376, resolution = 128):
    """
    AERS:
    Generates Golden Angle (GA) masks for a specified number of bars and resolution.

    This function creates a series of binary masks using the Golden Angle (GA) method, where each mask is 
    rotated by an increment determined by the golden angle. The function generates masks for a finite 
    number of GAs (default is 376), because GA of n=0 and n=377 are very similar.

    Parameters:
    -----------
    - num_bars : int, optional
        The number of golden angle masks to generate. 
            Default is 376.
    - resolution : int, optional
        The resolution of each mask, specifying both the height and width. 
            Default is 128.

    Returns:
    --------
    - torch.Tensor
        A tensor containing the generated golden angle masks with dimensions `(num_bars, resolution, resolution)`.

    Notes:
    ------
    - The generated masks are useful in applications where rotational symmetry or uniform sampling over angles is required.
    - The output tensor has dimensions `num_bars x resolution x resolution`, where each slice along the first dimension corresponds to a different angle generated using the Golden Angle method.

    ================================================================================================
    """
    ans = mask_theta(90, (1,resolution, resolution))
    angle = 90
    for i in range(1,num_bars):
        angle = (angle + (360*(GA))/(2*np.pi)) 
        ans = torch.cat((ans,mask_theta(angle, (1,resolution, resolution))))
    return ans # AERS: ans dimentions are 376 x resolution x resolution


def fetch_loss_function(loss_str, device, loss_params):
    """
    AERS:
    Returns the appropriate loss function based on the provided loss type string.

    This function selects and returns the specified loss function, potentially with additional 
    parameters and configurations. If the loss function is a combination of multiple losses, 
    it constructs the appropriate composite loss.

    Parameters:
    -----------
    - loss_str : str
        The type of loss function to return. Expected values include:

        - 'None' : No loss function, returns None.
        - 'Cosine' : Cosine similarity loss.
        - 'L1' : L1 loss (mean absolute error).
        - 'L2' : L2 loss (mean squared error).
        - 'SSIM' : Structural Similarity Index Measure loss.
        - 'MS_SSIM' : Multi-Scale Structural Similarity Index Measure loss.
        - 'Cosine-L1' : Combined Cosine similarity and L1 loss.
        - 'Cosine-L2' : Combined Cosine similarity and L2 loss.
        - 'Cosine-SSIM' : Combined Cosine similarity and SSIM loss.
        - 'Cosine-MS_SSIM' : Combined Cosine similarity and MS-SSIM loss.
    - device : torch.device
        The device on which the loss function will be computed (e.g., 'cpu' or 'cuda').
    - loss_params : dict
        A dictionary of parameters needed for specific loss functions, such as:

        - 'SSIM_window' : The window size for SSIM loss.
        - 'alpha_phase' : The weighting factor for phase in combined losses.
        - 'alpha_amp' : The weighting factor for amplitude in combined losses.

    Returns:
    --------
    - callable or None
        The requested loss function as a callable object, or None if 'None' is specified.

    Notes:
    ------
    - Composite losses like 'Cosine-L1' and 'Cosine-L2' are constructed using the `dual_fft_loss` function, which expects shifted Fourier transforms as input.
    - Ensure that `loss_params` contains all the required parameters for the selected loss function.

    ================================================================================================
    """    
    if loss_str == 'None':
        return None
    elif loss_str == 'Cosine':
        return lambda x,y: torch.acos(torch.cos(x - y)*(1 - EPS*10**3)).mean()
    elif loss_str == 'L1':
        return nn.L1Loss().to(device)
    elif loss_str == 'L2':
        return nn.MSELoss().to(device)
    elif loss_str == 'SSIM':
        return kornia.losses.SSIMLoss(loss_params['SSIM_window']).to(device)
    elif loss_str == 'MS_SSIM':
        return kornia.losses.MS_SSIMLoss().to(device)
    elif loss_str == 'Cosine-L1':
        # expects a shifted fourier transform
        return dual_fft_loss(
                    fetch_loss_function('Cosine', device, loss_params), 
                    fetch_loss_function('L1', device, loss_params), 
                    alpha_phase = loss_params['alpha_phase'], 
                    alpha_amp = loss_params['alpha_amp']
                        )
    elif loss_str == 'Cosine-L2':
        # expects a shifted fourier transform
        return dual_fft_loss(
                    fetch_loss_function('Cosine', device, loss_params), 
                    fetch_loss_function('L2', device, loss_params), 
                    alpha_phase = loss_params['alpha_phase'], 
                    alpha_amp = loss_params['alpha_amp']
                        )
    elif loss_str == 'Cosine-SSIM':
        # expects a shifted fourier transform
        return dual_fft_loss(
                    fetch_loss_function('Cosine', device, loss_params), 
                    fetch_loss_function('SSIM', device, loss_params), 
                    alpha_phase = loss_params['alpha_phase'], 
                    alpha_amp = loss_params['alpha_amp']
                        )
    elif loss_str == 'Cosine-MS_SSIM':
        # expects a shifted fourier transform
        return dual_fft_loss(
                    fetch_loss_function('Cosine', device, loss_params), 
                    fetch_loss_function('MS_SSIM', device, loss_params), 
                    alpha_phase = loss_params['alpha_phase'], 
                    alpha_amp = loss_params['alpha_amp']
                        )

def dual_fft_loss(phase_loss, amp_loss, alpha_phase = 1, alpha_amp = 1):
    """
    AERS:
    Combines phase and amplitude losses for complex-valued inputs using Fourier transforms.

    This function creates a composite loss function that applies separate loss functions to the phase and 
    amplitude components of complex-valued data. The phase and amplitude components are first extracted from 
    the real and imaginary parts of the input, and then the respective loss functions are applied and 
    combined using the specified weighting factors.

    Parameters:
    -----------
    - phase_loss : callable
        A loss function that will be applied to the phase components of the inputs.
    - amp_loss : callable
        A loss function that will be applied to the amplitude components of the inputs.
    - alpha_phase : float, optional
        A weighting factor for the phase loss. Default is 1.
    - alpha_amp : float, optional
        A weighting factor for the amplitude loss. Default is 1.

    Returns:
    --------
    - callable
        A composite loss function that can be applied to pairs of complex-valued tensors.

    Notes:
    ------
    - This function returns a callable that expects two inputs, which are complex-valued tensors with their last dimension containing the real and imaginary parts.
    - The returned function will first unbind the complex-valued inputs into their real and imaginary parts, convert these to amplitude and phase, and then apply the respective loss functions.
   
    ================================================================================================
    """
    def apply_func(x1,x2):
        real1, imag1 = torch.unbind(x1, -1)
        real2, imag2 = torch.unbind(x2, -1)
        amp1,phase1 = convert_cylindrical_to_polar(real1, imag1)
        amp2,phase2 = convert_cylindrical_to_polar(real2, imag2)
        return phase_loss(phase1, phase2)*alpha_phase + amp_loss(amp1, amp2)*alpha_amp
    return apply_func

def mask_theta(theta, size):
    """
    AERS:
    Generates a binary mask with a line drawn at a specific angle.

    This function creates a binary mask of a given size and draws a line at the specified angle `theta`. 
    The line is drawn using a custom algorithm that finds the closest approximation to the desired angle 
    within the specified mask dimensions.

    Parameters:
    -----------
    - theta : float
        The angle at which to draw the line, in degrees. The angle is taken modulo 180 to ensure it falls 
        within the range [0, 180).
    - size : tuple of int
        The dimensions of the output mask. Expected to be a tuple in the format `(channels, height, width)` 
        where `channels` is typically 1 for a 2D mask.

    Returns:
    --------
    - torch.Tensor
        A binary tensor of the specified size with a line drawn at the specified angle. The line is represented 
        by 1s in the tensor, and the rest of the tensor is filled with 0s.

    Notes:
    ------
    - The algorithm determines the best line to approximate the desired angle by calculating the cosine of the angle 
      and comparing it to possible lines within the grid.
    - The function handles cases where the line extends beyond the mask's boundaries by adjusting the start and end points.
    - The `draw_line` function is called to actually render the line on the binary mask.

    ================================================================================================
    """

    r = size[-2]
    c = size[-1]
    theta = theta%180
    minval = float('inf')
    for tx in range(-c+1,c):
        costheta = (tx) / ((r-1-0)**2 + (0-tx)**2)**0.5
        if np.abs(costheta - np.cos((theta*np.pi)/180)) < minval:
            startx = 0
            starty = r-1
            endx = tx
            endy = 0
            minval = np.abs(costheta - np.cos((theta*np.pi)/180))
    if endx < 0:
        endx += c-1
        startx += c-1
    for ty in range(2*r-1):
        if ty < r:
            costheta = (c-1) / ((r-1-ty)**2 + (0-c-1)**2)**0.5
        else:
            costheta = -(c-1) / ((r-1-ty)**2 + (0-c-1)**2)**0.5
        if np.abs(costheta - np.cos((theta*np.pi)/180)) < minval:
            startx = 0
            starty = r-1
            endx = c-1
            endy = ty
            minval = np.abs(costheta - np.cos((theta*np.pi)/180))
    if endy >= r:
        endy -= r-1
        starty -= r-1
    
    distfromtop = min(starty, endy)
    distfrombottom = min(r-1-starty, r-1-endy)
    distfromleft = min(startx, endx)
    distfromright = min(c-1-startx, c-1-endx)
    moveup = distfromtop // 2
    moveright = distfromright // 2
    moveup -= distfrombottom // 2
    moveright -= distfromleft // 2

    starty -= moveup
    endy -= moveup
    startx += moveright
    endx += moveright
    return draw_line(torch.zeros(size), torch.tensor([startx, starty]), torch.tensor([endx, endy]), torch.tensor([1.]))

def get_window_mask(theta_init = 0, window_size = 7, number_of_radial_views = 14, resolution = 128):
    """
    AERS:
    Generates a windowed binary mask based on radial views and an initial angle.

    This function creates a windowed binary mask, where each window corresponds to a different 
    set of angles, calculated based on the specified number of radial views and an initial angle. 
    The masks for each window are summed and then binarized.

    Parameters:
    -----------
    - theta_init : float, optional
        The initial angle (in degrees) for the first radial view. Default is 0.
    - window_size : int, optional
        The number of windows (or layers) in the mask. Default is 7.
    - number_of_radial_views : int, optional
        The number of radial views used to calculate angles for the mask. Default is 14.
    - resolution : int, optional
        The resolution of each window in the mask, specifying both the height and width. Default is 128.

    Returns:
    --------
    - torch.Tensor
        A binary tensor of shape `(window_size, resolution, resolution)` where each layer in the 
        first dimension corresponds to a different window with radial masks.

    Notes:
    ------
    - The function first calculates the angles for each radial view and then creates a binary mask for each combination of window and radial view.
    - The resulting mask for each window is the sum of masks generated for that window, and is then binarized using the sign function.

    ================================================================================================
    """
    window_thetas = []
    window_mask = torch.zeros(window_size, resolution, resolution)
    for i in range(number_of_radial_views):
        window_thetas.append(i*(180/number_of_radial_views))
    theta_translations = []
    for i in range(window_size):
        theta_translations.append((i*180)/(window_size*number_of_radial_views))

    for wi in range(window_size):
        for thetai in window_thetas:
            m = mask_theta(theta_init + thetai + theta_translations[wi] , (1,resolution,resolution))
            window_mask[wi,:,:] += m.squeeze()
    window_mask = torch.sign(window_mask)
    return window_mask

def fspecial_gauss(size, sigma):

    """Function to mimic the 'fspecial' gaussian MATLAB function
    """

    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return torch.FloatTensor(g/g.sum())

# AERS: get_coil_mask data gets stored somewhere (metadata?) else so that they are always the same for each patient
def get_coil_mask(theta_init = 0, n_coils = 8, resolution = 128):
    """
    AERS:
    Generates coil sensitivity masks based on Gaussian distributions for a specified number of coils.

    This function creates a set of coil sensitivity masks, where each mask represents the sensitivity pattern 
    of a coil in a parallel MRI system. The coils are evenly distributed around a circle, and each coil's sensitivity 
    is modeled as a Gaussian distribution centered at a specific location determined by the angle `theta_init`.

    Parameters:
    ------------
    - theta_init : float, optional
        The initial angle (in radians) where the first coil's sensitivity is centered. Default is 0.
    - n_coils : int, optional
        The number of coils for which sensitivity masks are generated. Default is 8.
    - resolution : int, optional
        The resolution of each coil mask, specifying both the height and width of the square mask. Default is 128.

    Returns:
    --------
    - torch.Tensor
        A tensor of shape `(n_coils, resolution, resolution)` containing the generated coil sensitivity masks, 
        where each mask is normalized between 0 and 1.

    Notes:
    ------
    - The function uses a Gaussian filter to model each coil's sensitivity, with the sigma value controlling the spread.
    - The masks are generated by placing the Gaussian distributions on a larger grid, then extracting and normalizing the subregion that corresponds to the desired resolution.
    
    ================================================================================================
    """

    # AERS: 
    # theta_init: location of where coils start
    # n_coils: number of coils
    # resolution: matrix size
  
    # assert(len(centres) == n_coils)

    # AERS: Selects center pixel
    centrex, centrey = 2*resolution, 2*resolution

    # AERS: Sigma and radius control the coil coverage
    sigma = resolution/(2.5)
    radius = resolution//2

    # AERS: Creates the Gaussian filter
    filter = fspecial_gauss(2*resolution, sigma)
    ans = torch.zeros(n_coils, resolution, resolution)

    # AERS: Get Gaussian mask centered at given coordinates centre_g_x and centre_g_y
    for i in range(n_coils):
        # Coils are evenly distributed around 2π
        theta = theta_init + ((i/n_coils)*((2*np.pi)))

        # AERS: Estimates the coordinates of the center of the coil based on theta and radius
        x_delta = int(radius*np.cos(theta))
        y_delta = int(radius*np.sin(theta))
        centre_g_x = centrex + x_delta
        centre_g_y = centrey + y_delta

        # AERS: Places the mask in a large grid, then extracts the subregion that first the original size and centers (based on radius)
        # Final coil mask is normalized 0 to 1        
        big_ans = torch.zeros((4*resolution,4*resolution))
        big_ans[centre_g_x-resolution:centre_g_x+resolution, centre_g_y-resolution:centre_g_y+resolution] = filter
        ans[i,:,:] = big_ans[centrex-radius:centrex+radius, centrey-radius:centrey+radius]
        ans[i,:,:] = ans[i,:,:] - (ans[i,:,:].min())
        ans[i,:,:] = ans[i,:,:] / (ans[i,:,:].max()+1e-8)

    return ans    

def save_coils(theta_init, n_coils = 4):
    """
    AERS: Generates and saves coil sensitivity masks as images.

    This function generates coil sensitivity masks using the `get_coil_mask` function and saves each mask 
    as a grayscale image file. The images are saved in the current directory with filenames `Coil_0.png`, 
    `Coil_1.png`, etc., corresponding to each coil.

    Parameters:
    -----------
    - theta_init : float
        The initial angle (in radians) where the first coil’s sensitivity is centered.
    - n_coils : int, optional
        The number of coil sensitivity masks to generate and save. Default is 4.

    Returns:
    --------
    - None
        This function saves images to the file system and does not return any value.

    Notes:
    ------
    - The generated images are saved in the PNG format, with the filename format `Coil_<index>.png`.
    - The function uses a grayscale colormap (`cmap='gray'`) when saving the images.

    """    
    coils = get_coil_mask(theta_init = theta_init, n_coils = n_coils)
    for ci in range(coils.shape[0]):
        plt.imsave('Coil_{}.png'.format(ci), coils[ci,:,:], cmap = 'gray')
