import torch
import sys
import json
from models.yolo import Model
import os
import numpy as np

def pt2pkl(mpath):
    '''
    mpath : path of model.pt
    '''
    
    net = torch.load(mpath,map_location='cpu')
    # type(net) == dict
    # dict_keys:
    # (['epoch', 'best_fitness', 'training_results', 'model', 'optimizer'])
    net = net['model']

    name = mpath.replace('.pt','.pkl')
    # torch.save(net.state_dict(),name)
    w_new = {}
    for k,v in net.state_dict().items():
        w_new[k] = v.cpu().detach().numpy().astype(np.float32)
    import pickle
    pickle.dump(w_new, open(name,"wb"), protocol=2)

    print("Save the .pkl model successfully...")

if __name__ == "__main__":
    pt2pkl("weights/yolov5s.pt")
