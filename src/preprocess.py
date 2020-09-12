
from cv2 import cv2
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder

IMAGE_DIR  = '../data/images'
all_image = sorted(os.listdir(image_dir))

for f in all_image:
    image_file = image_dir + f.replace('jpg', 'png')

img = cv2.imread(image_file)

cv2.imshow(f, img)
cv2.waitKey(0)

def tensorize_image(image_path, output_shape):
    pass





#image_path, list of strings (paths: ["data/images/img1.png", .., "data/images/imgn.png"] corresponds to n images to be trained each step)
#output_shape, list of integers (shape = (n1, n2): n1, n2 is width and height of the DNN model's input)

#The tensor should be in [batch_size, output_shape[0], output_shape[1], 3] shape.

tensorize_mask(mask_path, output_shape):

tensorize_mask [batch_size, width, height]

encoder = OneHotEncoder()
encoder.fit(copy_mask)

torch.Tensor()

batch_masks = []

batch_masks.append(copy_mask)

batch_masks = np.array(batch_masks)

mask_tensor = torch.Tensor(batch_masks)

print(mask_tensor.shape)

#The tensor should be in [batch_size, output_shape[0], output_shape[1], 2] shape.

#Your model will accept the input with [batch_size, output_shape[0], output_shape[1], 3] shape and the label with [batch_size, output_shape[0], output_shape[1], 2] shape.



