
# coding: utf-8

# In[1]:


# video.py
'''
The "video" script takes the list of the generated images in sequence and creates a video
'''

# File system and command line options library
import os
import argparse

# Video clip editor library
from moviepy.editor import ImageSequenceClip

# Supported image extensions
IMAGE_EXT = ['jpeg', 'gif', 'png', 'jpg']


# In[ ]:


def main():
    parser = argparse.ArgumentParser(description='Create driving video.')
    
    # Input argument for the folder containing source images
    parser.add_argument(
        'image_folder',
        type=str,
        default='',
        help='Path to image folder. The video will be created from these images.'
    )
    
    # Input argument for the frame rate (FPS) default to 60FPS if not specified
    parser.add_argument(
        '--fps',
        type=int,
        default=60,
        help='FPS (Frames per second) setting for the video.')
    args = parser.parse_args()

    # Convert file folder into list filtered for image file types
    image_list = sorted([os.path.join(args.image_folder, image_file)
                        for image_file in os.listdir(args.image_folder)])
    
    image_list = [image_file for image_file in image_list if os.path.splitext(image_file)[1][1:].lower() in IMAGE_EXT]

    # Two methods of naming output video to handle varying environemnts
    video_file_1 = args.image_folder + '.mp4'
    video_file_2 = args.image_folder + 'output_video.mp4'

    # Create the video clip from the sequence of images
    print("Creating video {} @FPS={} ...".format(args.image_folder, args.fps))
    clip = ImageSequenceClip(image_list, fps=args.fps)
    
    try:
        clip.write_videofile(video_file_1)
    except:
        clip.write_videofile(video_file_2)


if __name__ == '__main__':
    main()

