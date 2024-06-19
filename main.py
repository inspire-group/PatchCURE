import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn


import numpy as np 
import os 
import argparse

from tqdm import tqdm
import time

from utils.builder import get_data_loader,build_pcure_model,build_undefended_model
from fvcore.nn import FlopCountAnalysis


parser = argparse.ArgumentParser()

parser.add_argument("--model-dir",default='checkpoints',type=str,help="checkpoints directory")
parser.add_argument('--data-dir', default='data', type=str,help="data directory")
parser.add_argument('--dataset', default='imagenet',type=str,help="dataset name")
parser.add_argument("--model",default='vitsrf14x2_masked',type=str,help="model name; see checkpoints/readme.md for more details")
parser.add_argument("--patch-size",default=32,type=int,help="size of the adversarial patch")
parser.add_argument("--batch-size",default=4,type=int,help="batch size for inference")
parser.add_argument("--num-img",default=-1,type=int,help="the number of images for experiments; use the entire dataset if -1")
#parser.add_argument("--num-mask",default=-1,type=int,help="the number of mask used in double-masking")
parser.add_argument("--mask-stride",default=1,type=int,help="the mask stride (double-masking parameter)")
parser.add_argument("--alg",default='pcure',choices=['pcure','pg','cbn','mr'],help="algorithm to use. set to pcure to obtain main results")
parser.add_argument("--certify",action='store_true',help="do certification")
parser.add_argument("--runtime",action='store_true',help="analyze runtime")
parser.add_argument("--flops",action='store_true',help="analyze flops")
parser.add_argument("--memory",action='store_true',help="analyze memory usage")
parser.add_argument("--undefended",action='store_true',help="experiment with undefended models")

args = parser.parse_args()

print(args)

val_loader = get_data_loader(args)

if args.undefended:
	model = build_undefended_model(args)
else:
	model = build_pcure_model(args)
device = 'cuda' 
model = model.to(device)
model.eval()

cudnn.benchmark = False
cudnn.deterministic = True


correct_undefended = 0
correct = 0 
total = 0 
certify = 0
data_time = []
inference_time = []
certification_time = []
flops_total = 0
memory_allocated = []
memory_reserved = []
start_time = time.time()
with torch.no_grad():
	if args.runtime: # dry run
		data = torch.ones(args.batch_size,3,224,224).cuda()
		for i in range(3):
			model(data)

	for data,labels in tqdm(val_loader):

		total +=len(labels)

		# data loading
		if args.runtime:
			torch.cuda.synchronize()
			a = time.time()		

		data=data.to(device)
		labels = labels.to(device)

		if args.runtime:
			torch.cuda.synchronize()
			data_time.append(time.time()-a)
		
		if args.flops:
			flops = FlopCountAnalysis(model, data)
			flops_total +=flops.total()/1e9
			continue

		# inference
		if args.runtime:
			torch.cuda.synchronize()
			a = time.time()
		if args.memory: 
			torch.cuda.reset_peak_memory_stats()

		output = model(data)

		if args.memory: 
			memory_allocated.append(torch.cuda.max_memory_allocated()/1048576) #to MB
			memory_reserved.append(torch.cuda.max_memory_reserved()/1048576)
		if args.runtime:
			torch.cuda.synchronize()
			inference_time.append(time.time()-a)

		if args.undefended: 
			output = torch.argmax(output,dim=1) # logits vector -> prediction label

		correct += torch.sum(output==labels).item()

		# certification
		if args.certify:
			if args.runtime:
				torch.cuda.synchronize()
				a = time.time()
			
			results = model.certify(data,labels)

			if args.runtime:
				torch.cuda.synchronize()
				certification_time.append(time.time()-a)

			certify += torch.sum(results).item()

print(f'Clean Accuracy: {correct/total}')
if args.certify:
	print(f'Certified Robust Accuracy: {certify/total}')
if args.runtime:
	data_time = np.sum(data_time)/total
	inference_time = np.sum(inference_time)/total
	certification_time = np.sum(certification_time)/total
	print(f'Throughput: {1/(data_time+inference_time)} img/s')
	#print(f'Data loading time: {data_time}; per-image inference time: {inference_time}; per-image certification time: {certification_time}')
if args.flops:
	print(f'Average per-image FLOP count: {flops_total/total}')

if args.memory:
	print(f'Memory allocated (MB): allocated {np.mean(memory_allocated[:-1])}')
	#print(np.mean(memory_allocated[:-1]))
	#print(np.mean(memory_allocated[1:-1]))
	#print(np.mean(memory_reserved[:-1]))
	#print(np.mean(memory_reserved[1:-1]))
#print(f'Experiment run time : {(time.time()-start_time)/60},{(time.time()-start_time)/3600}')