## FREESPACE SEGMENTATION WITH FULLY CONVOLUTIONAL NEURAL NETWORKS

In this project we are aimed to detect drivable area using semantic segmentation (with python, pytorch, opencv etc. technologies).

![result_gif](https://i.hizliresim.com/Wb5lyW.jpg)


## Json2Mask

Define directories for the folder containing json files and for mask output folder.

import os, tqdm, json, cv2
import numpy as np

JSON_DIR = '../data/test_jsons'
MASK_DIR  = '../data/test_masks'

if not os.path.exists(MASK_DIR):
    os.mkdir(MASK_DIR)
    
Get json files names and sort them.

In [14]:
json_list = os.listdir(JSON_DIR)
json_list.sort()
Read json files and convert them to dictionary. Get image sizes for create empty mask. Define mask path using directory and json file name. Image name and mask name will same. In a for loop find freespace class and get exterior points in list. Draw filled polygon on empty mask using points. Save mask in dictionary.

In [15]:
for json_name in tqdm.tqdm(json_list):
    json_path = os.path.join(JSON_DIR, json_name)
    json_file = open(json_path, 'r')
    json_dict = json.load(json_file)

    mask = np.zeros((json_dict["size"]["height"], json_dict["size"]["width"]), dtype=np.uint8)
    
    mask_path = os.path.join(MASK_DIR, json_name[:-5])
    
    for obj in json_dict["objects"]:
        if obj['classTitle']=='Freespace':
            mask = cv2.fillPoly(mask, np.array([obj['points']['exterior']]), color=1)
            
    cv2.imwrite(mask_path, mask.astype(np.uint8))
100%|██████████| 476/476 [00:04<00:00, 116.13it/s]
After the json to mask step we can write mask on image for check. Also we will have more meaning images.

## Mask on image

Define directories for the folder containing masks, images, mask_on_images folder.

In [16]:
import os, cv2, tqdm
import numpy as np
from os import listdir
from os.path import isfile, join

MASK_DIR  = '../data/test_masks'
IMAGE_DIR = '../data/test_images'
IMAGE_OUT_DIR = '../data/test_masked_images2'

if not os.path.exists(IMAGE_OUT_DIR):
    os.mkdir(IMAGE_OUT_DIR)

Define a function for compare mask file names and image file names. For write mask on image, every image must have a mask. For a correct comparison, we need to get filename before file type

In [17]:
def image_mask_check(image_path_list, mask_path_list):
    for image_path, mask_path in zip(image_path_list, mask_path_list):
        image_name = image_path.split('/')[-1].split('.')[0]
        mask_name  = mask_path.split('/')[-1].split('.')[0]
        assert image_name == mask_name, "Image and mask name does not match {} - {}".format(image_name, mask_name)

Define a function for write mask on image. Firstly read all file names in the folders. After that sort both list by name for correct match. Give that lists as parameter to image_mask_check function.
In a for loop read image and mask with openCv change mask color.

In [18]:
def write_mask_on_image():

    image_file_names = [f for f in listdir(IMAGE_DIR) if isfile(join(IMAGE_DIR, f))]
    mask_file_names = [f for f in listdir(MASK_DIR) if isfile(join(MASK_DIR, f))]

    image_file_names.sort()
    mask_file_names.sort()

    image_mask_check(image_file_names,mask_file_names)

    for image_file_name, mask_file_name in tqdm.tqdm(zip(image_file_names, mask_file_names)):
        image_path = os.path.join(IMAGE_DIR, image_file_name)
        mask_path = os.path.join(MASK_DIR, mask_file_name)
        mask  = cv2.imread(mask_path, 0).astype(np.uint8)
        image = cv2.imread(image_path).astype(np.uint8)

        mask_image = image.copy()
        mask_image[mask == 1, :] = (255, 0, 125)
        opac_image = (image/2 + mask_image/2).astype(np.uint8)
        
        cv2.imwrite(join(IMAGE_OUT_DIR, mask_file_name), opac_image)
Call write_mask_on_image() function for check.

In [19]:
make uncomment for create images
#write_mask_on_image()
476it [01:39,  4.76it/s]


## Preprocess


We should to prepare data before use on model. Model accept data as a tensor format. We need to transform images and masks to tensor.

Define directories for the folder containing masks and images folder.

In [ ]:
import numpy as np
import cv2, torch, glob
import matplotlib.pyplot as plt
from torchvision import transforms as T
from PIL import Image

MASK_DIR = "../data/test_masks"
IMG_DIR = "../data/test_images"


## Image to tensor


This function take four parameters. First parameter image paths, it must be list. Second parameter is output shape of image. Third parameter is selection for gpu calculation. Fourth parameter is selection for augmenting (brightness, contrast and hue) images.
This function use torchVision transform for resizing, augmenting and converting to tensor. Tensor format is [n, n_ch, w, h]

In [ ]:
def tensorize_image(image_path, output_shape, cuda=False, augment=False):
    dataset = list()
    Transform = list()
    
    Transform.append(T.Resize(output_shape))
    if augment:
        Transform.append(T.ColorJitter(brightness=0.4, contrast=0.4, hue=0.06))
    Transform.append(T.ToTensor())
    Transform = T.Compose(Transform)

    for file_name in image_path:
        image = Image.open(file_name)
        image = Transform(image)

        dataset.append(image)

    tensor = torch.stack(dataset)
    if cuda:
        tensor = tensor.cuda()
    return tensor
    
    
## Mask to tensor

Firstly we need to define two functions for mask to tensor. First one is create encoded mask which is included class information.

## One hot encode

Mask is a grayscale image, it is include two colors black (0) and white (1 or 255). In this mask black represent background and white represent freespace. This function create labels for representing classes. [0,1] label for background, [1,0] label for freespace. Function is returns np ndarray, array format is (width, height, n classes)

In [ ]:
def one_hot_encoder(data, n_class):
    encoded_data = np.zeros((*data.shape, n_class), dtype=np.int)
    encoded_labels = [[0,1], [1,0]]
    
    for lbl in range(n_class):
        encoded_label = encoded_labels[lbl]                   
        numerical_class_inds = data[:,:] == lbl                            
        encoded_data[numerical_class_inds] = encoded_label 

    return encoded_data

Also define a function for decode encoded data which is converts class information to image. This function returns image list as a np ndarray. We will use for show model result as an image.

In [ ]:
def decode_and_convert_image(data, n_class):
    decoded_data_list = []
    decoded_data = np.zeros((data.shape[2], data.shape[3]), dtype=np.int)

    for tensor in data:
        for i in range(len(tensor[0])):
            for j in range(len(tensor[1])):
                if (tensor[1][i,j] == 0):
                    decoded_data[i, j] = 255
                else: 
                    decoded_data[i, j] = 0
        decoded_data_list.append(decoded_data)

    return decoded_data_list

This function for change np ndarray format [w, h, n_ch] to [n_ch, w, h]. We need to change format for model.

In [ ]:
def torchlike_data(data):
    n_channels = data.shape[2]
    torchlike_data = np.empty((n_channels, data.shape[0], data.shape[1]))
    for ch in range(n_channels):
        torchlike_data[ch] = data[:,:,ch]
    return torchlike_data

This function take three parameters. First parameter image paths, it must be list. Second parameter is output shape of image. Third parameter is selection for gpu calculation.
Firstly read and resize mask using openCv. After that encode mask with "one_hot_encoder()" function. Change array format with "torchlike_data()" function, because model get tensor like [n, n_ch, w, h] format. Finally convert np ndarray data to tensor with "torch.from_numpy()" function. Result tensor format is [n, n_classes, w, h]

In [ ]:
def tensorize_mask(mask_path, output_shape ,n_class, cuda=False):
    batch_masks = list()

    for file_name in mask_path:
        mask = cv2.imread(file_name, 0)
        mask = cv2.resize(mask, output_shape)
        # mask = mask / 255
        encoded_mask = one_hot_encoder(mask, n_class)  
        torchlike_mask = torchlike_data(encoded_mask) #[C,W,H]

        batch_masks.append(torchlike_mask)      
  
    batch_masks = np.array(batch_masks, dtype=np.int)
    torch_mask = torch.from_numpy(batch_masks).float()
    if cuda:
        torch_mask = torch_mask.cuda()
    return torch_mask
    
## Model

I prefer U-Net model in this project because when i searching best model for semantic segmentation i saw lots of projects used U-Net model. According to my research U-Net is good on detect small objects in picture, also this model usually used on medical diseases detection.
This is my first project on deep learning, i am learning lots of new things doing this project. That's why i preferred take a working UNet model. I tried to understand how is model and training process work.

U-Net Model

![result](https://i.hizliresim.com/vgpy47.png)

### Data Augmentation

The data in the Train data set was reproduced with augmentation at different angles and applying different contrast, brightness and hue values.

    for image in tqdm.tqdm(train_input_path_list):
	    img=Image.open(image)
	    color_aug = T.ColorJitter(brightness=0.4, contrast=0.4, hue=0.06)

	    img_aug = color_aug(img)
	    new_path=image[:-4]+"-1"+".png"
	    new_path=new_path.replace('image', 'augmentation')
	    img_aug=np.array(img_aug)
	    cv2.imwrite(new_path,img_aug)

The replicated data was added to the train data set;

    aug_size=int(len(aug_mask_path_list)/2)
	train_input_path_list=aug_path_list[:aug_size]+train_input_path_list+aug_path_list[aug_size:]
	train_label_path_list=aug_mask_path_list[:aug_size]+train_label_path_list+aug_mask_path_list[aug_size:]

Epoch number changed to 25.
A few examples of new data added after these processes are applied;

![augmentation](https://i.hizliresim.com/yWCi8U.jpg)


The model was retrained with the duplicated train data set. New loss values and graph;

![new_loss](https://i.hizliresim.com/o3MpQ5.jpg)


![new_graph](https://i.hizliresim.com/bpXW5M.jpg)


There is a good improvement in the results obtained after data augmentation;



![new3](https://i.hizliresim.com/wChRGJ.jpg)





