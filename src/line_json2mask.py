import numpy as np
import cv2
import json
import os
import tqdm 


#mask_dir'e mask dosyasının dosya yolunu yazdık
MASK_DIR  = '../data/maskline'
if not os.path.exists(MASK_DIR):#böyle bir dosya yolunda dosya yoksa 
    os.mkdir(MASK_DIR)#böyle bir dosya yolu olan dosya oluşturuyor


jsons=os.listdir('.../data/jsons')#ann klasörü içindeki json dosyalarının isimleriyle liste oluşturuldu


JSON_DIR = '../data/jsons'#dosyanın yolu değişkene atandı
for json_name in tqdm.tqdm(jsons):#json listesinin içindeki elemanlara ulaşıldı
    json_path = os.path.join(JSON_DIR, json_name)#okunacak dosya yolu birleştirildi
    json_file = open(json_path, 'r')#dosya okuma işlemi
    json_dict=json.load(json_file)#json dosyasının içindekiler dict veri tipine çevrildi
    mask=np.zeros((json_dict["size"]["height"],json_dict["size"]["width"]), dtype=np.uint8)
    
    mask_path = os.path.join(MASK_DIR, json_name[:-5])
    # her bir json dosyasından elde ettiğimiz dict'lerin içerisindeki objects key'lerinin value'leri listeye eklendi
    
    for obj in json_dict["objects"]:# json_dict icindeki objects ulaşıldı 
    
        if obj['classTitle']=='Solid Line':#classTitle==Solid olanlar bulundu 
            cv2.polylines(mask,[np.array([obj['points']['exterior']])],False,color=1,thickness=16)
            #np.zeros ile olusturdugumuz maskler json dosyalarımızın icersindeki point'lerin konumlarıyla dolduruldu 
        elif  obj['classTitle']=='Dashed Line':#classTitle==Dashed olanlar bulundu 
            cv2.polylines(mask,[np.array([obj['points']['exterior']])],False,color=2,thickness=16)
    cv2.imwrite(mask_path,mask.astype(np.uint8))#imwrite ile mask_path içerisine doldurulan maskeler yazdırıldı
