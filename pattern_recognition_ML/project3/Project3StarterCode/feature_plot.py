import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms, utils
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image
import json
import tqdm

from models import *

import torchvision.transforms as T
from pathlib import Path
import time
from random import randint

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.manifold import TSNE



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
    
def get_conv_layers(model):
    
    model_weights =[]
    #we will save the all conv layers in this list
    conv_layers = []# get all the model children as list
    model_children = list(model.children())#counter to keep count of the conv layers
    print("===")
    print(len(model_children))
    print(model_children)
    counter = 0#append all the conv layers and their respective wights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter+=1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            print("==========================")
            for j in range(len(model_children[i])):
                print(model_children[i][j])
                child = model_children[i][j]
                print(child)
                if type(child) == nn.Conv2d:
                    counter+=1
                    model_weights.append(child.weight)
                    conv_layers.append(child)
    print(f"Total convolution layers: {counter}")
    return conv_layers, model_weights

def get_feature_maps(model, conv_layers, img_name=None):
    # pass the image through all the layers
    results = [conv_layers[0](img)]
    for i in range(1, len(conv_layers)):
        # pass the result from the last layer to the next layer
        results.append(conv_layers[i](results[-1]))
    # make a copy of the `results`
    outputs = results
    
    # visualize 64 features from each layer 
    # (although there are more feature maps in the upper layers)
    for num_layer in range(len(outputs)):
        plt.figure(figsize=(32, 32))
        layer_viz = outputs[num_layer][0, :, :, :]
        layer_viz = layer_viz.data
        print(layer_viz.size())
        for i, filter in enumerate(layer_viz):
            if i == 64: # we will visualize only 8x8 blocks from each layer
                break
            plt.subplot(8, 8, i + 1)
            plt.imshow(filter.cpu(), cmap='gray')
            plt.axis("off")
        print(f"Saving layer {num_layer} feature maps...")
        plt.savefig(f"./outputs/layer_{num_layer}_{img_name}.png")
        #plt.show()
        plt.close()
        
def tsne_plot(model, transform, data_root, test_set, device, activation, save_dir='./tsne_plots', save_name=""):
    #data_root = "./data/Wallpaper"
    #test_set = "test"
    #train_dataset = ImageFolder(os.path.join(data_root, 'train'), transform=transform)
    test_dataset = ImageFolder(os.path.join(data_root, test_set), transform=transform)
    #train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    model.eval()
    feature_list = []
    with torch.no_grad():
        preds = []
        targets = []
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            pred = output.argmax(dim=1)
            targets.append(target.cpu().numpy())
            preds.append(pred.cpu().numpy())
            feature_list.append(activation['fc_1'].cpu().numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    feature_list = np.concatenate(feature_list)
    
    
    print(feature_list.shape)
    print(targets.shape)
    print(targets)
    
    
    #return targets, feature_list
    
    
    
    
    
    
    '''
    
    labels = []
    image_paths = []
    outputs = np.array([])
    for batch in test_loader:
        images = batch['image'].to(device)
        labels += batch['label']
        image_paths += batch['image_path']
 
        output = model.forward(images)
    
        current_outputs = output.cpu().numpy()
        features = np.concatenate((outputs, current_outputs))
    '''
    tsne = TSNE(n_components=2).fit_transform(feature_list)
    
    def scale_to_01_range(x):
        # compute the distribution range
        value_range = (np.max(x) - np.min(x))
    
        # move the distribution so that it starts from zero
        # by extracting the minimal value from all its values
        starts_from_zero = x - np.min(x)
    
        # make the distribution fit [0; 1] by dividing by its range
        return starts_from_zero / value_range
 
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)
    
    # initialize a matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    class_names = test_dataset.classes
    class_names = np.unique(targets)
    print(class_names)
    
    color = []
    n = len(class_names)
    for i in range(n):
        color.append('#%06X' % randint(0, 0xFFFFFF))
    # for every class, we'll add a scatter plot separately
    for label in class_names:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(targets) if l == label]
    
        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
    
        # convert the class color to matplotlib format
        #color = np.array(class_names[label], dtype=np.float64) / 255
    
        # add a scatter plot with the corresponding color and label
        ax.scatter(current_tx, current_ty, c=color[label], label=label)
    
    # build a legend using the labels we set previously
    ax.legend(loc='best')
    
    # finally, show the plot
    plt.savefig(os.path.join(save_dir, f"tsne_{save_name}"))
    plt.show()
    
        
if __name__ == "__main__":
    train_transform = T.Compose([
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
    
    transform = T.Compose([

        T.Resize((128, 128)),
        T.Grayscale(),
        T.ToTensor(),
        T.Normalize((0.5, ), (0.5, )),
    ])
    
    model_name = 'model_saved.pt'
    model_folder = "./results"
    
    test_set = 'test'
    tsne_save_name = f"{model_folder[2:]}_{test_set}"
    num_classes=17
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN2(input_channels=1, img_size=128, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(os.path.join(model_folder, 'Wallpaper', model_name)))
    
    # add hook
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook



    model.fc_1.register_forward_hook(get_activation('fc_1'))

    
    
    img = Image.open(Path(f'data/Wallpaper/test_challenge/P6') / f'P6_1.png')
    img = transform(img).to(device) 
    img = img.unsqueeze(0)
    



    
    tsne_plot(model, transform, data_root="./data/Wallpaper", test_set=test_set, device=device, activation=activation, save_name=tsne_save_name)
    '''
    conv_layers, model_weights = get_conv_layers(model)
    
    # we will save the conv layer weights in this list
    
    # take a look at the conv layers and the respective weights
    for weight, conv in zip(model_weights, conv_layers):
        #print(f"WEIGHT: {weight} \nSHAPE: {weight.shape}")
        print(f"CONV: {conv} ====> SHAPE: {weight.shape}")
    
    folder_names = ["CM", "CMM", "P1", "P2", "P3", "P3M1", "P4", "P4G", "P4M", "P6", "P6M", "P31M", "PG", "PGG", "PM", "PMG", "PMM"]

    for folder in folder_names:   
        img = Image.open(Path(f'data/Wallpaper/test_challenge/{folder}') / f'{folder}_1.png')
        img = transform(img).to(device)     
        img = img.unsqueeze(0)
        #print(img.size())
        get_feature_maps(model, conv_layers, folder)
    '''