#===================================================================================#
# Computational Intelligence Lab(CIL) ETHZ Sp19 road segmentation competition		#
# Team Member: Yongqi Wang, Zhi Ye, Jingyuan Ma										#
# Author: Jingyuan Ma																#
# Version: v1.0																		#	
# Description: In this file, main network structure is written in this file.		#
#			   This is version v1.0 with network constructed with plain Convolution #
#			   Convolution Dimension is 											#
#			   loss is returned if training;otherwise, classfication				#
#===================================================================================#
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.checkpoint import checkpoint
from torch.nn.modules.batchnorm import BatchNorm2d as BN2D

from config import config

def get():
	return Network_v1(config.num_classes,None,None)
	
class Network_v1(nn.Module):
	def __init__(self, out_planes, is_training, pretrained_model=None):
		super(Network_v1, self).__init__()
		conv_channel = 128
		self.layers = []
		#conv7x7 with stride 1 padding 1, increase dept
		self.conv1 = ConvBnRelu(1, conv_channel, 7, 1, 3, 
								has_bn=True, has_relu=True, has_bias=False, 
								norm_layer=norm_layer)

		##conv7x7 with stride 1 padding 1, downsampling by 2
		self.conv2 = ConvBnRelu(conv_channel, conv_channel, 7, 2, 3,
								has_bn=True,has_relu=True, has_bias=False, 
								norm_layer=norm_layer)

		#conv3x3 with stride 1 padding 1, increase depth
		self.conv3 = ConvBnRelu(conv_channel, conv_channel*2, 3, 1, 1,  	
								has_bn=True,has_relu=True, has_bias=False, 
								norm_layer=norm_layer)

		#conv3x3 with stride 1 padding 1, downsampling by 2
		self.conv4 = ConvBnRelu(conv_channel*2, conv_channel*2, 3, 2, 1,
					   has_bn=True,
					   has_relu=True, has_bias=False, norm_layer=norm_layer)
					   
		#conv1x1 with stride 1 padding 1, reduce depth for classification
		self.conv5 = ConvBnRelu(conv_channel*2, outplanes, 3, 2, 1,	
								has_bn=True,
								has_relu=True, has_bias=False, norm_layer=norm_layer)

		self.loss = nn.criterion = nn.CrossEntropyLoss(reduction='mean',
									ignore_index=255)
		
		self.layers.append(self.conv1,self.conv2,self.conv3,self.conv4,self.conv5)
		
	def forward(self, x, gt):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.conv5(x)

		if is_training:
			loss = self.loss(x,gt)
			return loss
			
		return F.log_softmax(x, dim=1)

class ConvBnRelu(nn.Module):
	def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
				 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
				 has_relu=True, inplace=True, has_bias=False):
		super(ConvBnRelu, self).__init__()
		self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
							  stride=stride, padding=pad,
							  dilation=dilation, groups=groups, bias=has_bias)
		self.has_bn = has_bn
		if self.has_bn:
			self.bn = norm_layer(out_planes, eps=bn_eps)
		self.has_relu = has_relu
		if self.has_relu:
			self.relu = nn.ReLU(inplace=inplace)

	def forward(self, x):
		x = self.conv(x)
		if self.has_bn:
			x = self.bn(x)
		if self.has_relu:
			x = self.relu(x)

		return x
		