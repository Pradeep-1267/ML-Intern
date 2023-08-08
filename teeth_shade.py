import cv2
import numpy as np
from skimage.color import rgb2lab
from keras.models import load_model



class shade_predictor:
    def __init__(self):
        path = "./my_model_final.h5"
        self.model = load_model(path)
        self.lst = ['Yay!, your teeth are white' , 'oops!, your teeth are slightly yellow' , 'Oh, NO!. Your teeth are yellow']


    def pre_process_image(self , image , target_size=(224, 224)):

        pad_color=(255, 255, 255)
        height, width, _ = image.shape
        _scale = min(target_size[0] / width, target_size[1] / height)
        new_width = int(width * _scale)
        new_height = int(height * _scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        padded_image = np.full((target_size[1], target_size[0], 3), pad_color, dtype=np.uint8)
        x_offset = (target_size[0] - new_width) // 2
        y_offset = (target_size[1] - new_height) // 2
        padded_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = image
        img = rgb2lab(padded_image)
        img = np.expand_dims(img, axis=0)
        return img

        

    def predict(self ,path ):

        image = cv2.imread(path)[...,::-1]
        img = self.pre_process_image(image)
        res = self.model.predict(img , verbose = 0)

        return [np.argmax(res) ,res]
    

