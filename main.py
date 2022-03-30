 # RE4017: Project 1 - Image Reconstruction from Sinogram
 
 # Charlie Gorey O'Neill - 18222803
 # Emmett Lawlor - 18238831
 # Christian Ryan - 18257356
 # Sean McTiernan - 18224385
 # Kacper Dudek - 18228798


import numpy as np
from PIL import Image
from skimage.transform import rotate    # Image Rotation
import scipy.fftpack as fft             # Fast Fourier Transform
import scipy.misc                       # Includes a package to save array as .png
import itertools                        # To combine mulitple list into single list of all combinations
from sys import argv

#-------------------------------------METHODS-----------------------------------------------------------
def fft_translate(r,g,b):
    r = scipy.fftpack.rfft(r, axis = 1)
    g = scipy.fftpack.rfft(g, axis = 1)
    b = scipy.fftpack.rfft(b, axis = 1)
    return r,g,b

###############################################

def ramp_filter(ffts):
    """Filter the projections using a ramp filter"""
    if type(ffts) ==  list:
        return [ramp_filter(channel) for channel in ffts]
    #Ramp filter a 2-d array of 1-d FFTs (1-d FFTs along the rows).
    ramp = np.floor(np.arange(0.5, ffts.shape[1]//2 + 0.1, 0.5))
    return ffts * ramp

###############################################

def inverse_fft_translate(operator):
    """Return to the spatial domain using inverse Fourier Transform"""
    if type(operator) == list:
        return [inverse_fft_translate(channel) for channel in operator]
    return fft.irfft(operator, axis=1)

###############################################

def radon(image, steps):
    """Build the Radon Transform using 'steps' projections of 'image'."""
    dTheta = -180.0 / steps # Angle increment for rotations.
    projections = [rotate(image, i*dTheta).sum(axis=0) for i in range(steps)]
    return np.vstack(projections)

###############################################

def build_laminogram(radonT):
    """Generate a laminogram by simple backprojection using the Radon Transform of an image, 'radonT'."""
    if type(radonT) == list:
        return [build_laminogram(channel) for channel in radonT]
    laminogram = np.zeros((radonT.shape[1],radonT.shape[1]))
    dTheta = 180.0 / radonT.shape[0]
    for i in range(radonT.shape[0]):
        temp = np.tile(radonT[i],(radonT.shape[1],1))
        temp = rotate(temp, dTheta*i)
        laminogram += temp
    return laminogram

###############################################

def single_channel_rescaling(input):
    """Function for single channel rescaling"""
    if type(input) == list:
        return [single_channel_rescaling(channel) for channel in input]
    Chi, Clo = input.max(), input.min()
    Cnorm = 255 * (input-Clo)/(Chi - Clo)
    return np.floor(Cnorm).astype('uint8')

###############################################

def crop_image(img):
    """Crops out the outer 30% of the image"""
    height, width = img.size
    perc = 0.30
    left = width/2 * perc
    top = height/2 * perc
    bottom = height - top
    right = width - left
    return img.crop((left, top, right, bottom))

###############################################

def reconstruction(sinogram, window="", ramp=True):
    """Perform all steps of the reconstruction"""
    # Step 1: Split sinogram into 3 channel (RGB) sinograms and convert to arrays
    r,g,b = Image.Image.split(sinogram)

    r = np.asarray(r)
    g = np.asarray(g)
    b = np.asarray(b)

    M = np.size(r, axis = 1)

    # Step 2: Perform windowing if required
    if window in ["Hann", "Hanning"]:
        r = np.hanning(M)*r
        g = np.hanning(M)*g
        b = np.hanning(M)*b
    elif window == "Hamming":
        r = np.hamming(M)*r
        g = np.hamming(M)*g
        b = np.hamming(M)*b

    # Step 3: Apply 1-d FFT to each row of sinogram - translate into freq domain
    r,g,b = fft_translate(r,g,b)

    # Step 4: Apply ramp filter if required
    if ramp:
        r, g, b = ramp_filter([r, g, b])

    # Step 5: Back to spatial (sinogram) domain via inverse FFT
    r, g, b = inverse_fft_translate([r, g, b])
    
    # Step 6: Image reconstruction via backprojection of radon transforms of filtered sinograms
    r, g, b = build_laminogram([r, g, b])

    # Step 7: Single-Channel image rescaling
    r, g, b = single_channel_rescaling([r, g, b])

    # Step 8: Combine channels, crop image and save as PNG
    image = np.dstack((r,g,b))
    image = crop_image(Image.fromarray(image))
    title = 'Non-' if not ramp else ''
    Image.Image.save(image, f"{window}_{title}Filtered.png")

#------------------------------------------MAIN-----------------------------------------------------------

if __name__ == "__main__":
    # Load sinogram
    try:
        sinogram = Image.open(argv[1])
    except IndexError:
        print("No file input in terminal, using sinogram.png instead")
        sinogram = Image.open('sinogram.png')
    except FileNotFoundError:
        print("No file with that name found, please try again")
        quit()

    # Lists of windows and ramp filter options
    windows = ["Hamming", "Hanning", "No Window"]
    ramps = [True, False]

    # Perform reconstruction on each combination of window and ramp filter
    for window, ramp in list(itertools.product(windows, ramps)):
        reconstruction(sinogram=sinogram, window=window, ramp=ramp)