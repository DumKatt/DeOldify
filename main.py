from deoldify import device
from deoldify.device_id import DeviceId
from deoldify.visualize import *

import fastai
from pathlib import Path
import torch

import warnings

def main():
    torch.backends.cudnn.benchmark=True
    #choices:  CPU, GPU0...GPU7
    device.set(device=DeviceId.GPU0)

    if not torch.cuda.is_available():
        print('GPU not available.')

    warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")
    colorizer = get_video_colorizer()

    source_url = 'https://www.reddit.com/r/nextfuckinglevel/comments/jyq24w/this_shot_from_the_movie_wings_1927_is_too_good/' #@param {type:"string"}
    render_factor = 21  #@param {type: "slider", min: 5, max: 40}
    watermarked = False #@param {type:"boolean"}
    if source_url is not None and source_url !='':
        video_path = colorizer.colorize_from_url(source_url, 'video.mp4', render_factor, watermarked=watermarked)
    else:
        print('Provide a video url and try again.')

