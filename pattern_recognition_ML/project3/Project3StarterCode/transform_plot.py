from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as T

import time



import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'




plt.rcParams["savefig.bbox"] = 'tight'
orig_img = Image.open(Path('data/Wallpaper/train/P6M') / 'P6M_3.png')
# if you change the seed, make sure that the randomly-applied transforms
# properly show that the image can be both transformed and *not* transformed!
torch.manual_seed(0)


def plot(imgs, with_orig=True, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    transform = T.Compose([
        #w,h = transforms.functional.get_image_size()
        # perform image augmentation
        
        T.RandomAffine(degrees=(0, 360)),#), translate=(0.25,0.25)) # translate image by a max range of [-64,64] pixels and rotate between [0◦, 360◦]
        T.CenterCrop((181, 181)), # Assuming image size is 256
        T.RandomCrop((181-8, 181-8), padding=False), #equivalent to translate
        T.RandomAffine(degrees=0, scale=(1,2)),
        # perform default transformations
        T.Resize((128, 128)),
        T.Grayscale(),
        T.ToTensor(),
        T.Normalize((0.5, ), (0.5, )),
        #T.RandomErasing(),
    ])
    
    seed = int(time.time())
    torch.manual_seed(seed)
    np.random.seed(seed)
    imgs = []
    for i in range(5):
        imgs.append(transform(orig_img)[0,:,:])
    plot(imgs)