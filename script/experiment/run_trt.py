import ctypes
import os
import random
import sys
import threading
import time

# import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable





INPUT_H = 224  #defined in decode.h
INPUT_W = 224
CONF_THRESH = 0.75
IOU_THRESHOLD = 0.4
np.set_printoptions(threshold=np.inf)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

class DeepMAR_trt(object):
    """
    description: A Retineface class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path):
        # Create a Context on this device,
        self.cfx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings

    def infer(self, input_image_path):
        threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.

        self.cfx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # Do image preprocess
        input_image = self.preprocess_image(
            input_image_path
        )
        a = time.time()
        # Copy input image to host buffer
        np.copyto(host_inputs[0], input_image.ravel())
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()
        # Here we use the first row of output in that batch_size = 1
        output = host_outputs[0]

        # Do postprocess
        #result_boxes, result_scores, result_landmark = self.post_process(
        #   output, origin_h, origin_w
        #)
        b = time.time()-a
        print(b)
        print(output)
        # Draw rectangles and labels on the original image

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()

    def preprocess_image(self, input_image_path):
        """
        description: Read an image from image path, resize and pad it to target size,
                     normalize to [0,1],transform to NCHW format.
        param:
            input_image_path: str, image path
        return:
            image:  the processed image
            image_raw: the original image
            h: original height
            w: original width
        """
        #image_raw = cv2.imread(input_image_path)
        #h, w, c = image_raw.shape

        # Calculate widht and height and paddings
        '''
        r_w = INPUT_W / w
        r_h = INPUT_H / h
        if r_h > r_w:
            tw = INPUT_W
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((INPUT_H - th) / 2)
            ty2 = INPUT_H - th - ty1
        else:
            tw = int(r_h * w)
            th = INPUT_H
            tx1 = int((INPUT_W - tw) / 2)
            tx2 = INPUT_W - tw - tx1
            ty1 = ty2 = 0
        '''

        # Resize the image with long side while maintaining ratio
        #image = cv2.resize(image_raw, (tw, th))
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # dataset 
        normalize = transforms.Normalize(mean=mean, std=std)
        test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,])
        img = Image.open(input_image_path).convert('RGB')
        #img = Image.open('./dataset/demo/demo_image_00.png').convert('RGB')
        img_trans = test_transform( img ) 
        img_trans = torch.unsqueeze(img_trans, dim=0)
        img_var = Variable(img_trans).cuda()
        img_var = to_numpy(img_var)
        # Pad the short side with (128,128,128)
        #image = cv2.copyMakeBorder(
        #    image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
        #)
        #image = image.astype(np.float32)

        # HWC to CHW format:
        # image -= (104, 117, 123)
        # image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        #image = np.expand_dims(img_var, axis=0)
        # Convert the image to row-major order, also known as "C order":
        #image = np.ascontiguousarray(image)
        return img_var


class myThread(threading.Thread):
    def __init__(self, func, args):
        threading.Thread.__init__(self)
        self.func = func
        self.args = args

    def run(self):
        self.func(*self.args)


if __name__ == "__main__":
    # load custom plugins,make sure it has been generated
    #PLUGIN_LIBRARY = "build/libdecodeplugin.so"
    #ctypes.CDLL(PLUGIN_LIBRARY)
    engine_file_path = "DeepMAR_resnet50.engine"

    retinaface = DeepMAR_trt(engine_file_path)
    input_image_paths = ["./dataset/demo/demo_image_00.png"]
    for i in range(10):
        for input_image_path in input_image_paths:
            # create a new thread to do inference
            thread = myThread(retinaface.infer, [input_image_path])
            thread.start()
            thread.join()

    # destroy the instance
    retinaface.destroy()