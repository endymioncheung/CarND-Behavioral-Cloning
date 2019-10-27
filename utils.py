#utils.py

import matplotlib.image as mpimg
import numpy as np
import cv2

# Set the random state to 0 (for debugging purpose)
np.random.seed(0)

# Nvidia end to end learning input image spec
rows, cols, ch = 66, 200, 3
nn_input_shape = ((rows, cols, ch))

# Steering angle correction
# used for left and right camera image augmentation
steering_angle_correction_factor = 0.2

# Load image and data
def load_image(img_path):
    '''
    Load RGB images from a file
    '''
    return mpimg.imread(img_path)

#Image Preprocessing

def crop(img):
    '''
    Crop out non-road sections of the image - the top (sky) and the bottom (engine cover)
    '''
    return img[60:-25,:,:]

def resize(img,width=cols,height=rows):
    '''
    Resize the image to the input shape used by the convolution neural network model
    '''
    return cv2.resize(img,(width,height),cv2.INTER_AREA)

def rgb2yuv(img):
    '''
    Convert the RGB image to YUV color space
    Y is the brightness channel
    '''
    return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

def preprocess_image(img):
    '''
    Preprocess image pipeline
    '''
    img = crop(img)
    img = resize(img)
    img = rgb2yuv(img)
    return img

def random_camera(data_dir,batch_sample,enable_random=True):
    '''
    Randomly choose an image from the center, left or right, and
    apply the steering angle correction for the camera offset
    '''
    camera_selected = np.random.choice(3)
    img_path = data_dir+'IMG/'+batch_sample[camera_selected].split('/')[-1]
    steering_angle = float(batch_sample[3])

    if not enable_random:
        return load_image(img_path), steering_angle

    if camera_selected == 1:
        # left camera image
        return load_image(img_path), steering_angle+steering_angle_correction_factor
    elif camera_selected == 2:
        # right camera image
        return load_image(img_path), steering_angle-steering_angle_correction_factor
    else:
        # center camera image
        return load_image(img_path), steering_angle

def random_flip(img, steering_angle):
    '''
    Random horizontal flip and invert the steering angles
    '''
    if np.random.rand() < 0.5:
        return np.fliplr(img), -steering_angle
    else:
        return img, steering_angle

def random_brightness(img):
    '''
    Random adjust the brightness of the image
    '''
    # Convert the image in the HSV colorspace, V is the brightness channel
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Limit the ratio of brightness to +/- 50%
    new_brightness_ratio = 1.0 + (np.random.rand() - 0.5)
    img_hsv[:,:,2] =  img_hsv[:,:,2] * new_brightness_ratio

    # Convert the image back to the RGB
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

def random_translate(img, steering_angle,px=2):
    '''
    Random translate the images in the x and y direction
    in the specified number of pixels
    '''
    height, width = img.shape[:2]

    # Generate random pixels translation in x and y directions
    fx,fy = np.random.randint(-px,px,2)
    steering_angle += fx * 0.2

    M = np.float32([[1,0,height-fx],
                    [0,1,width-fy]])
    translated_img = cv2.warpAffine(img,M,(width,height))
    return translated_img, steering_angle

def augument_data(img,steering_angle):
    '''
    Apply a set of random transforms to image and steering angle
    to generalize the dataset
    '''

    img, steering_angle = random_flip(img, steering_angle)
    img = random_brightness(img)
    # img, steering_angle = random_translate(img, steering_angle)

    return img, steering_angle
