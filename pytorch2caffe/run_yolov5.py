# -*- coding: utf-8 -*-  
import torch
import os, sys
import time
import math
from caffe.proto import caffe_pb2 as pb2
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import cv2
import numpy as np
from importlib import import_module
from torchvision import models

# CUDA_VISIBLE_DEVICES="0"  #Specified GPUs range
# os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

def load_filtered_state_dict(model, snapshot=None):
    # By user apaszke from discuss.pytorch.or
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model.load_state_dict(snapshot)

def get_flops(net, input_shape=(1, 3, 256, 192)):
    from flops_benchmark import add_flops_counting_methods, start_flops_count
    input = torch.ones(input_shape)
    input = torch.autograd.Variable(input)

    net = add_flops_counting_methods(net)
    net = net.train()
    net.start_flops_count()

    _ = net(input)

    return net.compute_average_flops_cost()/1e9/2

#TODO
#net = torch_net.build_model()
# net = models.resnet50(pretrained=False)
# net.eval()

from yolov5 import YOLOV5

net = YOLOV5(cfg = "yolov5s_basketball_test.yaml")
# pretrained_dict = torch.load('/home/pytorch2caffe/yolov5/best.pkl')
# pretrained_dict = np.load("weights.npy")
# import pickle
# pretrained_dict = pickle.load(open('/home/pytorch2caffe/yolov5/bestv2.pkl','rb'))
import pickle
pretrained_dict = pickle.loads(open("yolov5s.pkl", "rb").read())

state={}
for i,(k,v) in enumerate(pretrained_dict.items()):
    if 'num_batches_tracked' not in k and k in net.state_dict():
        k = k.replace('module.', '')    #module is saved model with paralle mode  
        state[k] = torch.from_numpy(v)

net.load_state_dict(state)
net.eval()
input_shape=(1, 3, 640, 640)
# net_flops = get_flops(net, input_shape)
# print("The Model's Total FLOPS is : {:.6f} G FLOPS".format(net_flops))
net.eval()

use_cuda = False

from ConvertModel import ConvertModel_caffe
print('Converting...')
text_net, binary_weights = ConvertModel_caffe(net, input_shape, softmax=False, use_cuda=use_cuda, save_graph=True)

import google.protobuf.text_format
with open('yolov5.prototxt', 'w') as f:
    f.write(google.protobuf.text_format.MessageToString(text_net))
with open('yolov5.caffemodel', 'w') as f:
    f.write(binary_weights.SerializeToString())

