import argparse
import os
import numpy as np
import cv2

# configurations

# region = (ymin, ymax, xmin, xmax)
regions = {'light': (63, 90, 292, 302),
          'red': (64, 70, 294, 300),
          'amber': (73, 79, 294, 300),
          'green': (83, 89, 295, 301)}

# reference histogram files
ref_red_file = './reference_histograms/ref_red.npy'
ref_amber_file = './reference_histograms/ref_amber.npy'
ref_green_file = './reference_histograms/ref_green.npy'

# number of bins used for histogram
# should be consistant with the reference histogram files
nbins = 32


def load_hist():
    """
    load and return the precalculated histograms
    """
    with open(ref_red_file, 'rb') as f:
        red_ref = np.load(f)
    
    with open(ref_amber_file, 'rb') as f:
        amber_ref = np.load(f)
    
    with open(ref_green_file, 'rb') as f:
        green_ref = np.load(f)
    
    return red_ref, amber_ref, green_ref


def load_image(image_path):
    """
    open and return image
    """
    if os.path.isfile(image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise TypeError("Input image {} could not be opened!")
    else:
        raise FileNotFoundError("Input image {} is not available!")
        
    return img


def calchist(img):
    '''
    Helper function to apply preprocessing and calculate histogram
    '''
    # applying gaussian smoothing
    img = cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT)
    
    # converting to grayscale before histogram calculation
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return cv2.calcHist([img], [0], None, [nbins], [0, 255])


def detect_traffic_lights(img, threshold, verbose=0):
    """
    Checking the traffic light conditions in current image 
    using a histogram comparison
    """
    # cropping the regions of interest
    ymin, ymax, xmin, xmax = regions['red']
    red = img[ymin:ymax, xmin:xmax]

    ymin, ymax, xmin, xmax = regions['amber']
    amber = img[ymin:ymax, xmin:xmax]
    
    ymin, ymax, xmin, xmax = regions['green']
    green = img[ymin:ymax, xmin:xmax]
    
    # ############################
    # Calculating histogram
    hist_red = calchist(red)
    hist_amber = calchist(amber)
    hist_green = calchist(green)
    
    # ############################
    # Reading the reference histograms
    red_ref, amber_ref, green_ref = load_hist()
    
    # ############################
    # Compare histograms with reference histograms
    # The lower the score, the better (ie more similar) when using method=3
    method = 3
    red_score = cv2.compareHist(red_ref, hist_red, method)
    amber_score = cv2.compareHist(amber_ref, hist_amber, method)
    green_score = cv2.compareHist(green_ref, hist_green, method)
    if verbose > 0:
        print('scores (the lower the score, more similar): ')
        print('\tred\t : {:.3f}\n\tamber\t : {:.3f}\n\tgreen\t : {:.3f}\n'.format(red_score, amber_score, green_score))
    # ############################
    
    # Rule based decision
    is_red = red_score < threshold
    is_amber = amber_score < threshold
    is_green = green_score < threshold
    
    if is_red:
        # priority is given to RED
        return 'RED'
    elif is_amber and not is_red and not is_green:
        return 'AMBER'
    elif is_green and not is_red and not is_amber:
        return 'GREEN'
    else:
        return 'UNKNOWN'
        
        
def detect(opt):
    """
    detects and returns the signal color
    """
    img = load_image(opt.image)
    signal = detect_traffic_lights(img, opt.hist_thresh, opt.verbose)
    return signal
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='', help='input image path')
    parser.add_argument('--hist-thresh', type=float, default=0.5, help='histogram comparison threshold')
    parser.add_argument('--verbose', type=int, default=0, help='verbose needed 0-only result,  1- +score, 2- +processing')
    
    opt = parser.parse_args()
    if opt.verbose == 2:
        print('Arguments:')
        print(opt)
        
    detected_light = detect(opt)
    
    print('Current Traffic Signal: ', detected_light)
    