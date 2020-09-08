import json
import os
import glob
import numpy as np
import cv2

from os import listdir
from os.path import isfile, join
filelist = [f for f in listdir('../data/ex_jsons') if isfile(join('../data/ex_jsons/', f))]

#print(filelist) dosya adlarını yazıyor.
JSON_DIR  = '../data/ex_jsons/'

for file_name in [file for file in os.listdir(JSON_DIR) if file.endswith('.json')]:
  with open(JSON_DIR + file_name) as json_file:
    data = json.load(json_file)
  
  #print(data) JSONların içeriğini dict e çeviriyor ve yazıyor.

  json_dict = data
  json_size = json_dict["size"] # yükseklik ve genişlik ölçüsünü alma


MASK_DIR = '../data/ex_masks'
if not os.path.exists(MASK_DIR):
    os.mkdir(MASK_DIR)


json_dict = data
json_objs = json_dict["objects"]

for obj in json_objs:
  obj_title = obj['classTitle']
  if obj_title == 'Freespace':
      json_dict = data
      json_objs = json_dict["points"]
      json_points = obj['exterior']
  else:
   continue

   #mask = np.zeros_like(image)
  mask = np.zeros((_image.width, _image.height, 3), np.uint8) # ??

  pts = np.array(points, np.int32)

  image = blackimage (size=h,w,3)
  image = cv2.fillpoly(image, np.array([pts]))

  cv2.fillPoly(mask,[pts],(255,255,255))
    
  cv2.imwrite(join(MASK_DIR,_json+".png"),mask)
    

