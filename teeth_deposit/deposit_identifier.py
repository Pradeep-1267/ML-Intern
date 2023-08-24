import numpy as np
import os
from PIL import Image, ImageDraw
import torchvision
import os
from transformers import AutoFeatureExtractor
from transformers import AutoModelForObjectDetection
import torch
import matplotlib.pyplot as plt




# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

output_dir = "./yolo_model/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





class teeth_deposit_identifier:
    def __init__(self):
        self.model = AutoModelForObjectDetection.from_pretrained(output_dir)
        self.model.to(device)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("hustvl/yolos-tiny", size=512, max_size=864)
        
        
    def rescale_bboxes(self , out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b
    
   
    def non_max_suppression(self, bboxes, probs, threshold):
    
  
        probs = [float(x) for x in probs]
        threshold = 1-threshold
        sorted_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    
        probs = [probs[i] for i in sorted_indices]
        bboxes = [bboxes[i] for i in sorted_indices]
        
        for i in range(len(probs)):
            if probs[i] > 0:  # Consider only boxes with non-zero probabilities
            
            # Compare with subsequent boxes
                for j in range(i + 1, len(bboxes)):
                    if probs[j] > 0:
                        iou = self.calculate_iou(bboxes[i], bboxes[j])
                        if iou > threshold:
                            probs[j] = -1  # Suppress overlapping box
                            
        
        pro = [probs[i] for i in range(len(probs)) if probs[i] > 0]
        box = [bboxes[i] for i in range(len(probs)) if probs[i] > 0]
        return [box , pro]
    
    
    def display_image(self , img , box , pro):
        plt.figure(figsize=(16,10))
        plt.imshow(img)
        ax = plt.gca()
        colors = COLORS * 100
        
        for p , (xmin , ymin , w , h) , c in zip( pro , box , colors):
            ax.add_patch(plt.Rectangle((xmin, ymin), w, h,
                                   fill=False, color=c, linewidth=3))

            text = f'{p:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
        ax.axis('off')
        plt.show()
        
        
    def calculate_iou( self , box1, box2):
    # Calculate IoU between box1 and box2
        x1_1, y1_1, w1, h1 = box1
        x2_1 = x1_1 + w1
        y2_1 = y1_1 + h1

        x1_2, y1_2, w2, h2 = box2
        x2_2 = x1_2 + w2
        y2_2 = y1_2 + h2

        x_intersection = max(x1_1, x1_2)
        y_intersection = max(y1_1, y1_2)
        x_union = min(x2_1, x2_2)
        y_union = min(y2_1, y2_2)

        intersection_area = max(0, x_union - x_intersection) * max(0, y_union - y_intersection)

        area_box1 = w1 * h1
        area_box2 = w2 * h2
        union_area = area_box1 + area_box2 - intersection_area

        iou = intersection_area / union_area if union_area > 0 else 0
        return iou
        
        
        
    def get_bounding_boxes(self , img , outputs , th_1  , th_2 , display = False):
        
        threshold = th_1

        probas = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > threshold
        # convert predicted boxes from [0; 1] to image scales
        boxes = self.rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), img.size)
        prob = probas[keep]
        pro = []
        box = []
        for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
            cl = p.argmax()
            box.append([xmin , ymin , xmax-xmin , ymax-ymin])
            pro.append(float(p[cl]))
        box , pro =  self.non_max_suppression(box , pro , th_2)
        
        if(display):
            self.display_image( img , box , pro)
        

        return { "probs" : pro , "bbox" : box }
        
        
            
        
    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)
        
        
        
    def pre_process(self , img):

        encoding = self.feature_extractor(images=img, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        pixel_values = pixel_values.unsqueeze(0).to(device)
        return pixel_values
        
        
    def predict(self , image_path , th_1 = 0.5 , th_2 = 0.7 , display = True):
        
        img = Image.open(image_path)
        pixel_values = self.pre_process(img)

        outputs = self.model(pixel_values=pixel_values)
        return self.get_bounding_boxes(img, outputs , th_1 = th_1 , th_2 = th_2 , display = display )