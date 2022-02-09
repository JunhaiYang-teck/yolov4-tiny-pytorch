import os
import json
import random
import numpy as np
import xml.etree.ElementTree as ET

from utils.utils import get_classes


classes_path        = 'model_data/voc_classes.txt'
classes, _      = get_classes(classes_path)

train_percent       = 0.9
VOCdevkit_path  = '/data/vptp/images'


if __name__ == "__main__":
    random.seed(0)
    print("Generate txt in ImageSets.")
    annotationfilepath     = os.path.join(VOCdevkit_path, 'para.json')
    saveBasePath    = os.path.join(VOCdevkit_path, '')
    with open(annotationfilepath) as f:
        annotation = json.load(f)
    
    item_number = len(annotation.items())
    index_list = np.random.binomial(1, train_percent, item_number)
    
    print("train and val size",item_number)
    print("train size", len([value for value in index_list if value == 1 ]))
    print("train size", len([value for value in index_list if value == 0 ]))
    ftrain      = open(os.path.join(saveBasePath,'train.txt'), 'w')  
    fval        = open(os.path.join(saveBasePath,'val.txt'), 'w')  
    
    for index, (key, value) in enumerate(annotation.items()):
        if key is not None and value['radius'] != 0:
            print(key, value['center_x'], value['center_y'], value['radius'])
            x_min = value['center_x'] - value['radius']
            x_min = x_min if x_min >= 0 else 0
            y_min = value['center_y'] - value['radius']
            y_min = y_min if y_min >= 0 else 0
            x_max = value['center_x'] + value['radius']
            x_max = x_max if x_max <= 648 else 648
            y_max = value['center_y'] + value['radius']
            y_max = y_max if y_max <= 486 else 486
            if index_list[index] == 1:
                ftrain.write(f"{os.path.join(VOCdevkit_path, key)} {x_min} {y_min} {x_max} {y_max} 0\n")
            else:
                fval.write(f"{os.path.join(VOCdevkit_path, key)} {x_min} {y_min} {x_max} {y_max} 0\n")

    ftrain.close()  
    fval.close()  
    print("Generate txt in ImageSets done.")
