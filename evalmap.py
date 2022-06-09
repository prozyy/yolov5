import os
import cv2
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import torch

device = select_device("0")
model = attempt_load("/zyy/Code/yolov5_myhub/runs/exp4/weights/best.pt", map_location=device)  # load FP32 model
model.eval()

if __name__ == "__main__":

    ROOT_DIR = "/data1/Data/basketball/YOLOAnno/BasketballDet"
    coco = COCO(os.path.join(ROOT_DIR,"basketball_float.json"))

    imageIDs = list(coco.imgToAnns.keys())
    print("num:",len(imageIDs))
    choosedImageIDs = []
    
    res = []
    for index,img_id in enumerate(imageIDs):
        if index % 100 == 0:
            print(index,"/",len(imageIDs))
        choosedImageIDs.append(img_id)
        imagePath = os.path.join(ROOT_DIR, "test_images",coco.loadImgs(img_id)[0]['file_name'])
        image = cv2.imread(imagePath)
        # image = cv2.imread(os.path.join(ROOT_DIR, "test_images","000015.jpg"))
        h,w,_ = image.shape

        if h / w > 384 / 640:
            ratio = 384 / h
        else:
            ratio = 640 / w

        image_resized = cv2.resize(image,None,fx = ratio,fy = ratio)
        h0,w0,_ = image_resized.shape
        image_resized = image_resized.transpose((2,0,1)).astype(np.float32)

        image_resized_ = image_resized.copy()
        image_resized[0,:,:] = image_resized_[2,:,:]
        image_resized[2,:,:] = image_resized_[0,:,:]

        image_input = np.zeros((1,3,384,640),dtype=np.float32)

        image_input[0,:,((384 - h0) // 2) : ((384 - h0) // 2 + h0),((640-w0) // 2):((640-w0) // 2 + w0)] = image_resized.copy()

        img = torch.from_numpy(image_input).to(device)
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img)[0]
        # Apply NMS
        pred = non_max_suppression(pred, 0.001, 0.65)

        if pred[0] is None:
            continue

        pred = pred[0].cpu().detach().numpy()

        pred[:,0] = (pred[:,0] - (640-w0) // 2) * w / w0
        pred[:,2] = (pred[:,2] - (640-w0) // 2) * w / w0

        pred[:,1] = (pred[:,1] - (384-h0) // 2) * h / h0
        pred[:,3] = (pred[:,3] - (384-h0) // 2) * h / h0

        for det in pred:
            x1 = np.min([np.max([det[0],0]),w])
            y1 = np.min([np.max([det[1],0]),h])
            x2 = np.min([np.max([det[2],0]),w])
            y2 = np.min([np.max([det[3],0]),h])
            x = x1
            y = y1
            h = y2 - y1
            w = x2 - x1

            conf = det[4]
            res.append([int(img_id),round(x, 6),round(y, 6),round(w, 6), round(h, 6),  round(conf,6),1])
        # break
    res = np.array(res)
    coco_pred = coco.loadRes(res)
    cocoEval = COCOeval(coco, coco_pred, "bbox")
    # cocoEval.params.imgIds = choosedImageIDs
    # cocoEval.params.catIds = [1,2]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
