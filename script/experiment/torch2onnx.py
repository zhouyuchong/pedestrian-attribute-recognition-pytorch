import sys
import os
from traceback import print_stack
sys.path.append(os.getcwd())

import numpy as np
import random
import math

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.nn.parallel import DataParallel
import pickle as pickle
import time
import argparse
from PIL import Image, ImageFont, ImageDraw

from baseline.model.DeepMAR import DeepMAR_ResNet50
from baseline.utils.utils import str2bool
from baseline.utils.utils import save_ckpt, load_ckpt
from baseline.utils.utils import load_state_dict 
from baseline.utils.utils import set_devices
from baseline.utils.utils import set_seed

def export_onnx(model, im, file, opset=12, train=False, dynamic=False, simplify=True):
    # ONNX export
    try:
        import onnx
        f = file

        torch.onnx.export(model, im, f, verbose=False, opset_version=opset,
                          training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=not train,
                          input_names=['input0'],
                          output_names=['output0'],
                          dynamic_axes={'input0': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
                                        'output0': {0: 'batch', 1: 'attrs'},  # shape(1,25200,85)
                                        } if dynamic else None)

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify
        if simplify:
            try:
                import onnxsim

                model_onnx, check = onnxsim.simplify(
                    model_onnx,
                    dynamic_input_shape=dynamic,
                    input_shapes={'input0': list(im.shape)} if dynamic else None)
                assert check, 'assert check failed'
                onnx.save(model_onnx, f)
            except Exception as e:
                print("simplify failed \n{}".format(e))
    except Exception as e:
        print('onnx->trt failure: \n{}'.format(e))

class Config(object):
    def __init__(self):
        
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--sys_device_ids', type=eval, default=(0,))
        parser.add_argument('--set_seed', type=str2bool, default=False)
        # model
        parser.add_argument('--resize', type=eval, default=(224, 224))
        parser.add_argument('--last_conv_stride', type=int, default=2, choices=[1,2])
        # demo image
        parser.add_argument('--demo_image', type=str, default='./dataset/demo/demo_image_00.png')
        ## dataset parameter
        parser.add_argument('--dataset', type=str, default='peta',
                choices=['peta','rap', 'pa100k', 'rap2'])
        # utils
        parser.add_argument('--load_model_weight', type=str2bool, default=True)
        parser.add_argument('--model_weight_file', type=str, default='./exp/deepmar_resnet50/peta/partition0/run1/model/petapt_epoch150.pth')
        args = parser.parse_args()
        
        # gpu ids
        self.sys_device_ids = args.sys_device_ids

        # random
        self.set_seed = args.set_seed
        if self.set_seed:
            self.rand_seed = 0
        else: 
            self.rand_seed = None
        self.resize = args.resize
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # utils
        self.load_model_weight = args.load_model_weight
        self.model_weight_file = args.model_weight_file
        if self.load_model_weight:
            if self.model_weight_file == '':
                print ('Please input the model_weight_file if you want to load model weight')
                raise ValueError
        # dataset 
        datasets = dict()
        datasets['peta'] = './dataset/peta/peta_dataset.pkl'
        datasets['rap'] = './dataset/rap/rap_dataset.pkl'
        datasets['rap2'] = './dataset/rap2/rap2_dataset.pkl'
        datasets['pa100k'] = './dataset/pa100k/pa100k_dataset.pkl'

        if args.dataset in datasets:
            dataset = pickle.load(open(datasets[args.dataset], 'rb'))
        else:
            print ('%s does not exist.'%(args.dataset))
            raise ValueError
        self.att_list = [dataset['att_name'][i] for i in dataset['selected_attribute']]
        
        # demo image
        self.demo_image = args.demo_image

        # model
        model_kwargs = dict()
        model_kwargs['num_att'] = len(self.att_list)
        model_kwargs['last_conv_stride'] = args.last_conv_stride
        self.model_kwargs = model_kwargs

### main function ###
cfg = Config()

# dump the configuration to log.
import pprint
print('-' * 60)
print('cfg.__dict__')
pprint.pprint(cfg.__dict__)
print('-' * 60)

# set the random seed
if cfg.set_seed:
    set_seed( cfg.rand_seed )
# init the gpu ids
set_devices(cfg.sys_device_ids)

# dataset 
normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)
test_transform = transforms.Compose([
        transforms.Resize(cfg.resize),
        transforms.ToTensor(),
        normalize,])

### Att model ###
model = DeepMAR_ResNet50(**cfg.model_kwargs)

# load model weight if necessary
if cfg.load_model_weight:
    map_location = (lambda storage, loc:storage)
    ckpt = torch.load(cfg.model_weight_file, map_location=map_location)
    model.load_state_dict(ckpt['state_dicts'][0])
model.cuda()
model.eval()

dummy_input = Variable(torch.randn(1, 3, 224, 224))
dummy_input = dummy_input.to('cuda')
# torch.onnx.export(model, dummy_input, "peta.onnx", opset_version = 12)
export_onnx(model=model, im=dummy_input, file="peta.onnx", dynamic=True, simplify=True)


