from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, sys
 
sys.path.append(os.getcwd())
import onnxruntime
import onnx
 

 
import warnings
 
import cv2
import onnx
import torch
import numpy as np
import onnxruntime
from PIL import Image
import torchvision.transforms as transforms
import pickle as pickle
from torch.autograd import Variable
import time
 
warnings.filterwarnings("ignore")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

 
class ONNXModel():
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))
 
    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name
 
    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name
 
    def get_input_feed(self, input_name, image_tensor):
        """
        input_feed={self.input_name: image_tensor}
        :param input_name:
        :param image_tensor:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_tensor
        return input_feed
 
    def forward(self, image_tensor):
        '''
        image_tensor = image.transpose(2, 0, 1)
        image_tensor = image_tensor[np.newaxis, :]
        onnx_session.run([output_name], {input_name: x})
        :param image_tensor:
        :return:
        '''
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_tensor})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: image_tensor})
        input_feed = self.get_input_feed(self.input_name, image_tensor)
        scores = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return scores

datasets = dict()
datasets['peta'] = './dataset/peta/peta_dataset.pkl'

dataset = pickle.load(open(datasets['peta'], 'rb'))
    
att_list = [dataset['att_name'][i] for i in dataset['selected_attribute']]
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
# dataset 
normalize = transforms.Normalize(mean=mean, std=std)
test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,])


label_idx = list(np.arange(0, 7))
worker = ONNXModel("./peta.onnx")
 
# img = cv2.imread("./dataset/demo/demo_image_00.png")
# load one image 
img = Image.open('./dataset/demo/demo_image_00.png').convert('RGB')
img_trans = test_transform( img ) 
img_trans = torch.unsqueeze(img_trans, dim=0)
img_var = Variable(img_trans).cuda()
s_time = time.time()
output = worker.forward(to_numpy(img_var))
print(type(output))
e_time = time.time()
print("time usage: ", (e_time - s_time))
output = output[0]

# show the score in command line
for idx in range(58):
    #print("inded {} label {} score {}".format(idx, cfg.att_list[idx], score[0, idx]))
    if output[0, idx] >= 0:
        print ('%s: %.2f'%(att_list[idx], output[0, idx]))
'''
# show the score in the image
img = img.resize(size=(256, 512), resample=Image.BILINEAR)
draw = ImageDraw.Draw(img)
positive_cnt = 0
for idx in range(len(cfg.att_list)):
    if score[0, idx] >= 0:
        txt = '%s: %.2f'%(cfg.att_list[idx], score[0, idx])
        draw.text((10, 10 + 10*positive_cnt), txt, (255, 0, 0))
        positive_cnt += 1
path = cfg.demo_image[:-4] + '_' + cfg.model_weight_file[-6:-4] + '_result.png'
img.save(path)
'''