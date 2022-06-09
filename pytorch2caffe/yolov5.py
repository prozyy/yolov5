import torch
import torch.nn as nn
import yaml
import math
from copy import deepcopy
from commonConvert import Conv, ConvT, Bottleneck, SPP, Focus, BottleneckCSP,Concat
import numpy as np

def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor

def parse_model(d, ch):  # model_dict, input_channels(3)
    print('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv,ConvT, nn.Conv2d, Bottleneck, SPP, Focus, BottleneckCSP]:
            c1, c2 = ch[f], args[0]

            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            args = [c1, c2] + list(args[1:])
            if m in [BottleneckCSP]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
        elif m is Detect:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]
        args = [int(a) if isinstance(a,float) else a for a in args ]
        print("args:",args,"type: ",str(m))
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        print('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

class Detect(nn.Module):
    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors

    def forward(self, x):
        z = []
        for i in range(self.nl):
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            # x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            z.append(x[i].sigmoid()) 
            print("z",z[-1].shape)

        return z

class YOLOV5(nn.Module):
    def __init__(self, cfg='yolov5s_person_test.yaml', ch=3, nc=None):
        super(YOLOV5, self).__init__()
        with open(cfg) as f:
            self.yaml = yaml.load(f)  # model dict

        # Define model
        if nc and nc != self.yaml['nc']:
            print('Overriding model.yaml nc=%g with nc=%g' % (self.yaml['nc'], nc))
            self.yaml['nc'] = nc  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist, ch_out
        initialize_weights(self)

    def forward(self,x):
        y, dt = [], []  # outputs
        for index,m in enumerate(self.model):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        return x

def load_filtered_state_dict(model, snapshot=None):
    # By user apaszke from discuss.pytorch.or
    model_dict = model.state_dict()
    # snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    new_snapshot = {}
    for k, v in snapshot.items():
        k = k.replace('module.', '')    #module is saved model with paralle mode  
        if k in model_dict:
            new_snapshot[k] = v
    if len(new_snapshot) == 0: print("bug...")
    model.load_state_dict(new_snapshot)

if __name__ == "__main__":
    net = YOLOV5()
    net.eval()
    # load_filtered_state_dict(net, snapshot=torch.load("../runs/exp0/weights/best.pkl"))
    import pickle
    pretrained_dict = pickle.loads(open("best.pkl", "rb").read())
    state={}
    for i,(k,v) in enumerate(pretrained_dict.items()):
        if 'num_batches_tracked' not in k and k in net.state_dict():
            k = k.replace('module.', '')    #module is saved model with paralle mode  
            state[k] = torch.from_numpy(v)
    net.load_state_dict(state)
    net.eval()

    img = torch.ones([1,3,640,640])
    img[:,0,:,:] *= 0.1
    img[:,1,:,:] *= 0.2
    img[:,2,:,:] *= 0.3
    res = net(img)
    for index,r in enumerate(res):
        print(r.cpu().detach().numpy().shape)
        np.save(str(index),r.cpu().detach().numpy())






