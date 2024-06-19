from math import ceil
from utils.pcure import PatchCURE,SecureLayer,SecurePooling,PatchGuardPooling,CBNPooling,MRPooling,MRPC
from utils.bagnet import BAGNET_FUNC
from utils.vit_srf import vit_base_patch16_224_srf,vit_large_patch16_224_srf
from utils.split import split_resnet50_like,split_vit_like
from timm.models import load_checkpoint

import os 

import torch
import torch.nn as nn
from torchvision import datasets

import timm
from timm.data.transforms_factory import create_transform
from timm.data import IMAGENET_DEFAULT_STD, IMAGENET_DEFAULT_MEAN

import numpy as np 


def get_data_loader(args):
	DATA_DIR=os.path.join(args.data_dir,args.dataset)
	#MAE and BagNet cfg
	if args.dataset == 'imagenet':
		data_cfg = DEFAULT_CFG = {'input_size': (3, 224, 224), 'interpolation': 'bicubic', 'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'crop_pct': 224/256}#, 'crop_mode': 'center'} 
		ds_transform = create_transform(**data_cfg)
		val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR,'val'),ds_transform) 
	else:
		data_cfg = {'input_size': (3, 224, 224), 'interpolation': 'bicubic', 'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'crop_pct': 1}#, 'crop_mode': 'center'} #
		ds_transform = create_transform(**data_cfg)
		if args.dataset == 'cifar':
			val_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=ds_transform)
		else:
			raise NotImplementedError("only support imagenet for now")


	np.random.seed(233333333)#random seed for selecting test images
	if args.num_img>0: #select a random subset of args.num_img for experiments
		idxs=np.arange(len(val_dataset))
		np.random.shuffle(idxs)
		idxs=idxs[:args.num_img]
		val_dataset = torch.utils.data.Subset(val_dataset, idxs)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,shuffle=False,num_workers=4)
	return val_loader


def build_pcure_model(args):
	# build and initialize model
	# examples of model name
	# hybrid (SRF+LRF): vitsrf14x2_split9_vanilla
	# SRF-only: vitsrf14x2_masked
	# LRF-only: mae_vanilla
	MODEL_DIR=os.path.join('.',args.model_dir)
	split_point = -1 # default value --> SRF-only or LRF-only mode 
	MODEL_NAME = args.model.split('_')[:-1]
	if args.dataset != 'imagenet':
		MODEL_NAME = MODEL_NAME[:-1] # remove the suffix for dataset name
	if len(MODEL_NAME) == 1: # LRF-only or SRF-only
		MODEL_NAME = MODEL_NAME[0]
		split_point = -1 
	elif len(MODEL_NAME) == 2:
		MODEL_NAME, split_point = MODEL_NAME[0], int(MODEL_NAME[1][5:])
	else:
		raise NotImplementedError('model name not supported')

	lrf = None 
	srf = None

	if args.dataset == 'imagenet': num_classes = 1000 
	if args.dataset == 'cifar': num_classes = 10

	#init SRF if needed 
	if 'bagnet' in MODEL_NAME: #bagnet
		model_func = BAGNET_FUNC[MODEL_NAME]
		srf = model_func(pretrained=False,avg_pool=False)
		if split_point<0: # 
			rf_size = int(MODEL_NAME[6:]) 
			rf_size = (rf_size,rf_size)
			rf_stride = (8,8)
		else:
			raise NotImplementedError('only support SRF-only mode for BagNet') # for hybrid, bagnet RF needs to be changed accordingly 
	elif 'vitsrf' in MODEL_NAME or 'vitlsrf' in MODEL_NAME: # ViT-SRF -- vitsrf for ViT-Base; vitlsrf for ViT-Large
		# parse window size
		i = MODEL_NAME.find('srf')+3
		window_size = MODEL_NAME[i:].split('x')
		window_size = [int(x) for x in window_size]
		# build vit_srf model
		srf = vit_base_patch16_224_srf(window_size=window_size,return_features=True) if 'vitsrf' in MODEL_NAME else vit_large_patch16_224_srf(window_size=window_size,return_features=True)
		rf_size = (window_size[0]*16,window_size[1]*16)
		rf_stride = rf_size 

	# init LRF if needed 
	if 'resnet50' in MODEL_NAME or ('bagnet' in MODEL_NAME and split_point>0):
		lrf = timm.create_model('resnet50')
		lrf.reset_classifier(num_classes=num_classes)
	elif 'mae' in MODEL_NAME or ('vitsrf' in MODEL_NAME and split_point>=0):
		lrf = timm.create_model('vit_base_patch16_224',global_pool='avg') #the MAE setup
		lrf.reset_classifier(num_classes=num_classes)

	# calculate the corruption size in the secure layer
	patch_size = (args.patch_size,args.patch_size)
	# corruption_size: the "patch size" in the SRF feature map (for secure operation)
	# feature_size: the size of the SRF feature map (for secure operation)
	# mask_stride, mask_size: mask strides and sizes in the feature map
	# Note: for LRF-only (PatchCleanser), the secure operation layer is the input image
	# for now patch size and mask stride are **manually** set.
	
	data_cfg = {'input_size': (3, 224, 224)} # hard-coded image size for ImageNet

	if srf:
		corruption_size = ceil((patch_size[0] + rf_size[0] -1) / rf_stride[0]),ceil((patch_size[1] + rf_size[1] -1) / rf_stride[1]) #"patch size" in the feature space

		# dry run to get the feature map size
		srf.eval()
		dummy_img = torch.zeros(data_cfg['input_size']).unsqueeze(0)
		feature_shape = srf(dummy_img).shape

		feature_size = (feature_shape[-2],feature_shape[-1])
		mask_stride = (args.mask_stride,args.mask_stride)

	else: # pure patchcleanser
		corruption_size = patch_size
		feature_size = (data_cfg['input_size'][-2],data_cfg['input_size'][-1])
		mask_stride = (args.mask_stride,args.mask_stride) # num_mask -> mask_stride mapping: {6:33,5:39,4:49,3:65,2:97}

	# calculate mask size
	mask_size = (min(corruption_size[0]+mask_stride[0]-1,feature_size[0]),min(corruption_size[1]+mask_stride[1]-1,feature_size[1]))

	print('patch_size',patch_size)
	print('corruption_size',corruption_size)
	print('feature_size',feature_size)
	print('mask_size',mask_size)
	print('mask_stride',mask_stride)


	# construct PatchCURE model and load weights
	checkpoint_name = args.model + '.pth.tar' 
	checkpoint_path = os.path.join(MODEL_DIR,checkpoint_name) 

	if split_point<0: # SRF-only or LRF-only
		model = srf or lrf
		if lrf: 
			load_checkpoint(lrf,checkpoint_path)#,remap=False)
			secure_layer = SecureLayer(lrf,input_size=feature_size,mask_size=mask_size,mask_stride=mask_stride) 
			model = PatchCURE(nn.Identity(),secure_layer) # no SRF --> use nn.Indentify for the SRF sub-model
		elif srf:
			load_checkpoint(srf,checkpoint_path)#,remap=False)
			if args.alg=='cbn':# clipped bagnet
				secure_layer = CBNPooling()
			elif args.alg=='pg': #patchguard
				secure_layer = PatchGuardPooling(mask_size=mask_size)
			elif args.alg=='pcure': # pcure srf-only
				secure_layer = SecurePooling(input_size=feature_size,mask_size=mask_size,mask_stride=mask_stride)
			elif args.alg == 'mr':
				secure_layer = MRPooling(input_size=feature_size,mask_size=mask_size,mask_stride=mask_stride)
			else:
				raise NotImplementedError
			model = PatchCURE(srf,secure_layer)
	else: #hybrid
		if 'vitsrf' in args.model: 
			srf,_ = split_vit_like(srf,split_point,True) # get the first half of SRF
			_,lrf = split_vit_like(lrf,split_point) # get the second half of LRF
			lrf.num_window = srf.num_window
			lrf.num_patch = srf.num_patch
			lrf.window_size = srf.window_size
		elif 'bagnet' in args.model:
			raise NotImplementedError('only support SRF-only mode for BagNet') 
		# combine SRF and LRF together to instantiate PatchCURE
		if args.alg == 'pcure':
			secure_layer = SecureLayer(lrf,input_size=feature_size,mask_size=mask_size,mask_stride=mask_stride) 
		elif args.alg == 'mr':
			secure_layer = MRPC(lrf,input_size=feature_size,mask_size=mask_size,mask_stride=mask_stride) 
		else:
			raise NotImplementedError

		model = PatchCURE(srf,secure_layer)
		# need to create PatchCURE instance before load_checkpoint.
		# need to remap the state_dict *key*. (the names of weight tensors are a bit different; I used nn.Sequential(srf,lrf) during training)
		load_checkpoint(model,checkpoint_path,remap=True) 
	return model 

def build_undefended_model(args):

	MODEL_DIR=os.path.join('.',args.model_dir)
	MODEL_NAME = args.model.split('_')[0]
	if 'bagnet' in MODEL_NAME: #bagnet
		model_func = BAGNET_FUNC[MODEL_NAME]
		model = model_func(pretrained=False,avg_pool=True)
	elif 'resnet50' in MODEL_NAME:
		model = timm.create_model('resnet50',pretrained=True)
		#model=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
	elif 'mae' in MODEL_NAME :
		model = timm.create_model('vit_base_patch16_224',global_pool='avg') #the MAE setup
	elif 'vit' in MODEL_NAME:
		i = MODEL_NAME.find('srf')+3
		window_size = MODEL_NAME[i:].split('x')
		window_size = [int(x) for x in window_size]
		model = vit_base_patch16_224_srf(window_size=window_size,return_features=False) if 'vitsrf' in MODEL_NAME else vit_large_patch16_224_srf(window_size=window_size,return_features=False)
	# load model
	checkpoint_name = args.model + '.pth.tar' 
	try:
		checkpoint_path = os.path.join(MODEL_DIR,checkpoint_name) 
		load_checkpoint(model,checkpoint_path)
	except:
		print('local checkpoints not loaded; might have initilize with pretrained weights')
	return model