import pickle
pretrained_dict = pickle.loads(open("yolov5s1.pkl", "rb").read())


for i,(k,v) in enumerate(pretrained_dict.items()):
    if "detect" in k:
        print(k)