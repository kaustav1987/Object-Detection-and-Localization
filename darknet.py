##imports
from __future__ import division
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
from util import predict_transform
import warnings
warnings.filterwarnings('ignore')

def get_test_img(file_path):
    img = cv2.imread(file_path)
    img = cv2.resize(img,(416,416))

    ##img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W X C -> C X H X W
    img =  img[np.newaxis,:,:,:]
    img = img/255.0
    img_ = torch.from_numpy(img).float()
    img = Variable(img_)
    return img


def parse_cfg(cfg_file):
    """
    Takes a configuration file

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    """
    with open(cfg_file,'r') as file:
        lines = file.read().split('\n')
        lines = [x for x in lines if len(x)>0]
        lines = [x for x in lines if x[0]!= '#']
        lines = [x.rstrip().lstrip() for x in lines]

    block ={}
    blocks =[]

    for line in lines:
        if line[0]== '[':           # This marks the start of a new block
            if len(block) !=0:
                blocks.append(block)
                block = {}
            block['type']= line[1:-1].rstrip()
        else:
            key , value = line.split('=')
            block[key.rstrip()] = value.rstrip()

    blocks.append(block)

    return blocks

def create_modules(blocks):
    '''
    create module using the blocks
    '''
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()
        if x['type']== 'convolutional':
            #If it's an convolutional layer
            activation = x['activation']
            try:
                batch_normalize = int(x['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x['filters'])
            stride = int(x['stride'])
            padding = int(x['pad'])
            kernel_size = int(x['size'])
            if padding >0:
                pad = (kernel_size-1)//2
            else:
                pad = 0
            #Add the convolutional layer
            conv = nn.Conv2d(prev_filters,filters, kernel_size,stride, pad,bias = bias)
            module.add_module("conv_{0}".format(index),conv)
            #Add the batch Normalization layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("bn_{0}".format(index),bn)
            #Check the activation
            #It is either Linear or a Leaky ReLU for YOLO
            if activation == 'Relu':
                actvn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), actvn)

        elif    x['type']== 'upsample':
            #If it's an upsampling layer
            #We use Bilinear2dUpsampling
            stride = int(x['stride'])
            upsample = nn.Upsample(scale_factor = stride, mode ="nearest")
            module.add_module("upsample_{}".format(index),upsample)

        elif x['type']== 'route':
            #If it is a route layer
            x['layers']= x['layers'].split(',')
            start = int(x['layers'][0])
            try:
                end = int(x['layers'][1])
            except:
                end = 0
            #Positive anotation
            if start >0:
                start = start - index
            if end >0:
                end = end - index

            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)

            if end <0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        elif x['type'] == 'shortcut':
            #shortcut corresponds to skip connection
            shortcut = EmptyLayer()
            module.add_module('shortcut_{}'.format(index), shortcut)

        elif x['type'] == 'yolo':
            #Yolo is the detection layer
            mask = x['mask'].split(',')
            mask = [int(x) for x in mask]
            anchors = x['anchors'].split(',')
            anchors =[int(x) for x in anchors]
            #anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
            # 9 anchors each has 2 values , width and height
            anchors = [(anchors[i], anchors[i+1]) for i in range(0,len(anchors),2)]
            anchors = [anchors[i] for i in mask]
            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        module_list.append(module)
        prev_filters= filters
        output_filters.append(filters)
    return (net_info,module_list)

## The below class is for Route and shortcut layer
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

## The below class is for the achor selection when type is 'yolo'
class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

#blocks = parse_cfg("cfg/yolov3.cfg")
#print(create_modules(blocks))

## Declaring the Darket class
class DarkNet(nn.Module):
    def __init__(self, cfg_file):
        super(DarkNet, self).__init__()
        self.blocks = parse_cfg(cfg_file)
        self.net_info, self.module_list= create_modules(self.blocks)

    def forward(self, x , cuda):
        """Forward Pass"""
        modules = self.blocks[1:]
        outputs = {}
        write = 0
        for i, module in enumerate(modules):
            module_type = module['type']
            if module_type== 'convolutional' or module_type == 'upsample':
                x = self.module_list[i](x)

            elif module_type== 'route':
                layers = module['layers']
                layers =[int(x) for x in layers]
                if layers[0] > 0:
                    layers[0] = layers[0]- i
                if len(layers) ==1:
                    x = outputs[i+ layers[0]]
                else:
                    if layers[1] > 0:
                        layers[1]= layers[1]-i
                    map0 = outputs[i +layers[0]]
                    map1 = outputs[i + layers[1]]
                    x = torch.cat((map0, map1), dim=1)

            elif module_type == 'shortcut':
                from_ = int(module["from"])
                #print(outputs[i-1].shape)
                #print(outputs[i+ from_].shape)
                x = outputs[i-1] + outputs[i+ from_]

            elif module_type=='yolo':
                anchors = self.module_list[i][0].anchors
                #Get the input dimensions
                inp_dim = int(self.net_info['height'])
                #Get the number of classes
                num_classes = int(module['classes'])

                #Transform
                x = x.data
                cuda_available = torch.cuda.is_available()
                x = predict_transform(x, inp_dim,anchors,num_classes,cuda_available)
                if not write:              #if no collector has been intialised.
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections,x),dim=1)

            outputs[i] = x
        return detections
    def load_weights(self, weight_file):
        fp = open(weight_file, 'rb')
        #The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype=np.int32, count =5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype = np.float32)
        ptr = 0
        for i in range(len(self.module_list)):
            model_type = self.blocks[i+1]
            if model_type == 'convolution':
                model_layer = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]['batch_normalize'])
                except:
                    batch_normalize = 0
                conv = model_layer[0]
                if batch_normalize:
                    bn = model_layer[1]
                    #Get the number of weights of Batch Norm Layer
                    num_bn_bias = bn.bias.numel()
                    #Load the bn_weights
                    bn_bias = torch.from_numpy(weights[ptr:ptr+ num_bn_bias])
                    ptr = ptr + num_bn_bias
                    bn_weights = torch.from_numpy(weights[ptr:ptr+ num_bn_bias])
                    ptr = ptr + num_bn_bias
                    bn_running_mean  = torch.from_numpy(weights[ptr:ptr+ num_bn_bias])
                    ptr = ptr + num_bn_bias
                    bn_running_var = torch.from_numpy(weights[ptr:ptr+ num_bn_bias])
                    ptr = ptr + num_bn_bias
                    #Cast the loaded weights into dims of model weights.
                    bn_bias = bn_bias.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
                    #Copy the data to model
                    bn.bias.data.copy_(bn_bias)
                    bn.weights.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                else:
                    #Number of biases
                    num_bias = conv.bias.numel()

                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr:ptr +num_bias])
                    ptr = ptr + num_bias
                    conv_biases.view_as(conv.bias.data)

                    conv.bias.data.copy_(conv_biases)

                #Let us load the weights for the Convolutional layers
                num_conv = conv.weight.numel()
                conv_weights = torch.from_numpy(weights[ptr:ptr+ num_conv])
                ptr = ptr + num_conv
                conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)




model = DarkNet("cfg/yolov3.cfg")
model.load_weights("yolov3.weights")
path = 'dog.jpg'
inp   = get_test_img(path)
pred = model(inp,torch.cuda.is_available())
print(pred)
print('Shape:', pred.shape)
