import torch
import time
import torch.nn.functional as F
import numpy as np

class PatchCURE(torch.nn.Module):
	def __init__(self,srf,lrf):
		super().__init__()
		self.srf = srf
		self.lrf = lrf
	def forward(self,x):
		x = self.srf(x)
		x = self.lrf(x)
		return x
	def certify(self,x,y,thres=None):
		x = self.srf(x)
		return self.lrf.certify(x,y)


class SecurePooling(torch.nn.Module):
	# for SRF-only
	def __init__(self,input_size,mask_size,mask_stride):
		super().__init__()
		print(input_size,mask_size,mask_stride)
		self.mask_stride = mask_stride
		self.mask_size = mask_size
		self.input_size = input_size
		self._generate_mask(input_size,mask_size,mask_stride)

	def _generate_mask(self,input_size,mask_size,mask_stride):
		# set self.mask_list -- the binary mask tensors
		# set self.mask_coordinate_list # may not be used, depending on how we implement the inference algorithm
		mask_x_list = list(range(0,input_size[0] - mask_size[0] + 1,mask_stride[0]))
		if (input_size[0] - mask_size[0])%mask_stride[0]!=0:
			mask_x_list.append(input_size[0] - mask_size[0])
		mask_y_list = list(range(0,input_size[1] - mask_size[1] + 1,mask_stride[1]))
		if (input_size[1] - mask_size[1])%mask_stride[1]!=0:
			mask_y_list.append(input_size[1] - mask_size[1])

		self.mask_coordinate_list = torch.empty(len(mask_x_list),len(mask_y_list),2,dtype=int)
		self.mask_list = torch.ones(len(mask_x_list),len(mask_y_list),1,input_size[0],input_size[1]).cuda()# float32 masking should be faster than bool
		for mask_i,mask_x in enumerate(mask_x_list):
			for mask_j,mask_y in enumerate(mask_y_list):
				self.mask_coordinate_list[mask_i,mask_j,0] = mask_x
				self.mask_coordinate_list[mask_i,mask_j,1] = mask_y
				self.mask_list[mask_i,mask_j,:,mask_x:mask_x+mask_size[0],mask_y:mask_y+mask_size[1]] = 0
		print('num of masks',len(mask_x_list),len(mask_y_list))


	def _get_one_mask_logits(self,x,return_unmasked=True):
		# get prediction logits for all different one-mask predictions
		unmasked_logits = torch.sum(x,dim=(2,3),keepdim=True) # [B,C,1,1]
		window_sum = F.avg_pool2d(x,kernel_size=self.mask_size, stride=self.mask_stride, divisor_override=1)# [B,C,H',W']
		one_mask_logits = unmasked_logits - window_sum # [B,C,H',W']
		unmasked_logits = unmasked_logits if return_unmasked else None 
		return one_mask_logits,unmasked_logits  # unmasked_logits is the prediction logits vector without any masking
		
	def forward(self,x):
		# x [B,C,H,W]
		one_mask_logits,unmasked_logits = self._get_one_mask_logits(x)
		one_mask_pred = torch.argmax(one_mask_logits,dim=1) # [B,H',W']
		pred = torch.argmax(unmasked_logits,dim=1) #[B,1,1]
		##TODO: write another function using unique return_counts, then the certification is independendt on the vanilla prediction
		# now we just check aggrement with base predicion on unmasked image
		one_mask_disagreement_map = one_mask_pred != pred # [B,H',W'] 
		one_mask_disagreement = torch.any(one_mask_disagreement_map.flatten(start_dim=1),dim=-1)
		pred = pred.flatten()

		for img_i in torch.nonzero(one_mask_disagreement).view(-1):
			disagreers_i,disagreers_j = torch.nonzero(one_mask_disagreement_map[img_i],as_tuple=True)
			mask =  self.mask_list[disagreers_i,disagreers_j] # [1,1,H,W]
			tmp = mask * x[img_i].unsqueeze(0)
			two_mask_logits,_ = self._get_one_mask_logits(tmp) # [N,C,H',W']
			two_mask_pred = torch.argmax(two_mask_logits,dim=1).view(len(disagreers_i),-1) #
			one_mask_disagreer = one_mask_pred[img_i,disagreers_i,disagreers_j]
			two_mask_agreement = one_mask_disagreer.unsqueeze(-1) == two_mask_pred
			two_mask_agreement = torch.all(two_mask_agreement,dim=1)
			two_mask_agreed = one_mask_disagreer[two_mask_agreement]
			if len(two_mask_agreed)>0:
				pred[img_i] = two_mask_pred[:,0][two_mask_agreement][0]
		return pred

		'''
		# another equivalent version without for loop # more compact but less readable. easier to get OOM.
		disagreers_imgi,disagreers_i,disagreers_j = torch.nonzero(one_mask_disagreement_map,as_tuple=True) # each [N,]
		if len(disagreers_i)>0: 
			mask = self.mask_list[disagreers_i,disagreers_j] # [N,1,H,W]
			tmp = x[disagreers_imgi] * mask # [N,C,H,W]
			two_mask_logits,_ = self._get_one_mask_logits(tmp) # [N,C,H',W']
			two_mask_pred = torch.argmax(two_mask_logits,dim=1).flatten(start_dim=1)
			one_mask_disagreer = one_mask_pred[disagreers_imgi,disagreers_i,disagreers_j]
			two_mask_agreement = one_mask_disagreer.unsqueeze(-1) == two_mask_pred
			two_mask_agreement = torch.all(two_mask_agreement,dim=1)
			pred[disagreers_imgi[two_mask_agreement]] = one_mask_disagreer[two_mask_agreement]
		return pred
		
		'''

	def certify(self,x,y):
		results = torch.zeros_like(y,dtype=bool)
		one_mask_logits,_ = self._get_one_mask_logits(x,return_unmasked=False)
		one_mask_pred = torch.argmax(one_mask_logits,dim=1) # [B,H',W']
		one_mask_correctness = torch.all(one_mask_pred.flatten(start_dim=1) == y.unsqueeze(-1),dim=1) # only check two-mask correctness when there is one-mask correctness
		mask =  self.mask_list.view(-1,1,self.input_size[0],self.input_size[1]) # [N,1,H,W]
		for img_i in torch.nonzero(one_mask_correctness).flatten():
			tmp = mask * x[img_i].unsqueeze(0)
			two_mask_logits,_ = self._get_one_mask_logits(tmp,return_unmasked=False) # [N,C,H',W']
			two_mask_pred = torch.argmax(two_mask_logits,dim=1)
			results[img_i] = torch.all(two_mask_pred==y[img_i])
		return results



class SecureLayer(SecurePooling): 
	# for LRF-only and LRF+SRF
	def __init__(self,lrf,input_size,mask_size,mask_stride):
		super().__init__(input_size,mask_size,mask_stride)
		self.lrf = lrf 

	def _get_one_mask_logits(self,x,return_unmasked=True):
		unmasked_logits = self.lrf(x).unsqueeze(-1).unsqueeze(-1) if return_unmasked else None # [B,C,1,1] 
		one_masked_image = torch.einsum('bcij,nmcij -> bnmcij',x,self.mask_list) #[B,N1,N2,C,W,H]
		B,N1,N2,C,W,H = one_masked_image.shape
		one_masked_image = one_masked_image.flatten(end_dim=2) #[B*N1*N2,C,W,H]
		one_mask_logits = self.lrf(one_masked_image).view([B,N1,N2,-1]).permute(0,3,1,2)
		return one_mask_logits,unmasked_logits

	def certify(self,x,y):
		# ps: the certify() from the SecurePooling class also gives the same reuslts, but can be much just slower...
		results = torch.zeros_like(y,dtype=bool)
		mask_list =  self.mask_list.view(-1,1,self.input_size[0],self.input_size[1]) # [N,1,H,W]
		num_mask = len(mask_list)
		num_img = len(y)
		prediction_map = torch.zeros([num_img,num_mask,num_mask],dtype=int,device = x.device)
		for i,mask in enumerate(mask_list):
			for j in range(i,num_mask):
				mask2 = mask_list[j]
				masked_output = self.lrf(mask*mask2*x)
				_, masked_pred = masked_output.max(1)
				prediction_map[:,i,j] = masked_pred
		for img_i in range(num_img):
			pred_map = prediction_map[img_i]
			pred_map = pred_map + pred_map.T - torch.diag(torch.diag(pred_map))
			results[img_i] = torch.all(pred_map==y[img_i])
		return results


# below are some other implementation not used in main.py 
# e.g., model = PatchCURE(BagNet(),PatchGuardPooling())
class PatchGuardPooling(torch.nn.Module):
	def __init__(self,mask_size):
		super().__init__()
		self.mask_size = mask_size

	def forward(self,x):
		x = torch.clamp(x,min=0)
		unmasked_logits = torch.sum(x,dim=(2,3),keepdim=True) # [B,C,1,1]
		window_sum = F.avg_pool2d(x,kernel_size=self.mask_size, stride=(1,1), divisor_override=1)# [B,C,H',W']
		max_window_sum,_ = torch.max(window_sum.flatten(2),dim=2) # [B,C]
		masked_logits = unmasked_logits.squeeze() - max_window_sum
		#return masked_logits
		return torch.argmax(masked_logits,dim=1)
	def certify(self,x,y):
		# being lazy here, copied PatchGuard's implementation (which uses numpy and might be slow)
		x = torch.clamp(x,min=0).permute(0,2,3,1).detach().cpu().numpy()
		y = y.cpu().numpy()
		certify = torch.tensor([self._provable_masking(x[i],y[i],window_shape=self.mask_size)==2 for i in range(len(y))])
		return certify


	def _provable_masking(self,local_feature,label,clipping=-1,thres=0.,window_shape=[6,6],ds=False):
		'''
		local_feature	numpy.ndarray, feature tensor in the shape of [feature_size_x,feature_size_y,num_cls]
		label 			int, true label
		clipping 		int/float, the positive clipping value ($c_h$ in the paper). If clipping < 0, treat clipping as np.inf
		thres 			float in [0,1], detection threshold. ($T$ in the paper)
		window_shape	list [int,int], the shape of sliding window
		ds 				boolean, whether is for mask-ds

		Return 		int, provable analysis results (0: incorrect clean prediction; 1: possible attack found; 2: certified robustness )
		'''

		feature_size_x,feature_size_y,num_cls = local_feature.shape
		window_size_x,window_size_y = window_shape
		num_window_x = feature_size_x - window_size_x + 1 if not ds else feature_size_x
		num_window_y = feature_size_y - window_size_y + 1 if not ds else feature_size_y

		if clipping > 0:
			local_feature = np.clip(local_feature,0,clipping)
		else:
			local_feature = np.clip(local_feature,0,np.inf)

		global_feature = np.sum(local_feature,axis=(0,1))

		pred_list = np.argsort(global_feature,kind='stable')
		global_pred = pred_list[-1]

		if global_pred != label: # clean prediction is incorrect
			return 0

		local_feature_pred = local_feature[:,:,global_pred]

		# the sum of class evidence within each window
		in_window_sum_tensor = np.zeros([num_window_x,num_window_y,num_cls])

		for x in range(0,num_window_x):
			for y in range(0,num_window_y):
				if ds and x+window_size_x>feature_size_x:  #only happens when ds is True
					in_window_sum_tensor[x,y,:] = np.sum(local_feature[x:,y:y+window_size_y,:],axis=(0,1)) + np.sum(local_feature[:x+window_size_x-feature_size_x,y:y+window_size_y,:],axis=(0,1))
				else:
					in_window_sum_tensor[x,y,:] = np.sum(local_feature[x:x+window_size_x,y:y+window_size_y,:],axis=(0,1))


		idx = np.ones([num_cls],dtype=bool)
		idx[global_pred]=False
		for x in range(0,num_window_x):
			for y in range(0,num_window_y):

				# determine the upper bound of wrong class evidence
				global_feature_masked = global_feature - in_window_sum_tensor[x,y,:] # $t$ in the proof of Lemma 1
				global_feature_masked[idx]/=(1 - thres) # $t/(1-T)$, the upper bound of wrong class evidence 

				# determine the lower bound of true class evidence
				local_feature_pred_masked = local_feature_pred.copy()
				if ds and x+window_size_x>feature_size_x:
					local_feature_pred_masked[x:,y:y+window_size_y]=0 
					local_feature_pred_masked[:x+window_size_x-feature_size_x,y:y+window_size_y]=0 
				else:
					local_feature_pred_masked[x:x+window_size_x,y:y+window_size_y]=0 # operation $u\odot(1-w)$

				in_window_sum_pred_masked = in_window_sum_tensor[:,:,global_pred].copy()
				overlap_window_max_sum = 0
				# only need to recalculate the windows the are partially masked
				for xx in range(max(0,x - window_size_x + 1),min(x + window_size_x,num_window_x)):
					for yy in range(max(0,y - window_size_y + 1),min(y + window_size_y,num_window_y)):
						if ds and xx+window_size_x>feature_size_x:
							in_window_sum_pred_masked[xx,yy]=local_feature_pred_masked[xx:,yy:yy+window_size_y].sum()+local_feature_pred_masked[:xx+window_size_x-feature_size_x,yy:yy+window_size_y].sum()
							overlap_window_max_sum = in_window_sum_pred_masked[xx,yy] if overlap_window_max_sum<in_window_sum_pred_masked[xx,yy] else overlap_window_max_sum
						else:
							in_window_sum_pred_masked[xx,yy]=local_feature_pred_masked[xx:xx+window_size_x,yy:yy+window_size_y].sum()
							overlap_window_max_sum = in_window_sum_pred_masked[xx,yy] if overlap_window_max_sum<in_window_sum_pred_masked[xx,yy] else overlap_window_max_sum
							
				max_window_sum_pred = np.max(in_window_sum_pred_masked) # find the window with the largest sum
				if max_window_sum_pred / local_feature_pred_masked.sum() > thres: 
					global_feature_masked[global_pred]-=max_window_sum_pred
				else:
					global_feature_masked[global_pred]-=overlap_window_max_sum
					

				# determine if an attack is possible
				if np.argsort(global_feature_masked,kind='stable')[-1]!=label: 
					return 1

		return 2 #provable robustness



class CBNPooling(torch.nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self,x):
		x = torch.tanh(x*0.05-1)
		x =  torch.mean(x,dim=(2,3))
		return torch.argmax(x,dim=1)

	def certify(self,x,y):
		raise NotImplementedError()

class MRPooling(SecurePooling): # for now, for simplicty, I inherit the SecurePooling class. 
	def __init__(self,input_size,mask_size,mask_stride,thres=0):
		super().__init__(input_size,mask_size,mask_stride)
		self.C = input_size[0]*input_size[1] - mask_size[0]*mask_size[1]
		self.thres = thres

	def helper(self,x):
		one_mask_logits,_ = self._get_one_mask_logits(x)
		one_mask_logits = one_mask_logits/self.C 
		one_mask_softmax = torch.softmax(one_mask_logits,dim=1)
		one_mask_softmax = one_mask_softmax.flatten(start_dim=2)
		one_mask_conf,one_mask_pred = one_mask_softmax.max(dim=1)
		return one_mask_conf,one_mask_pred
	def forward(self,x):

		#one_mask_logits,unmasked_logits = self._get_one_mask_logits(x)
		#unmasked_pred_correct = torch.argmax(unmasked_logits.squeeze(),dim=1) == y 
		#one_mask_logits = one_mask_logits/self.C 
		#one_mask_softmax = torch.softmax(one_mask_logits,dim=1)
		#one_mask_softmax = one_mask_softmax.flatten(start_dim=2)
		#one_mask_conf,one_mask_pred = one_mask_softmax.max(dim=1)

		one_mask_conf,one_mask_pred = self.helper(x)
		#y = y.unsqueeze(-1)
		#correct = torch.logical_and(unmasked_pred_correct,~torch.any(torch.logical_and(one_mask_conf>self.thres,one_mask_pred!=y),dim=1))
		#certify = torch.logical_and(unmasked_pred_correct,torch.all(torch.logical_and(one_mask_conf>self.thres,one_mask_pred==y),dim=1))
		#correct = ~torch.any(torch.logical_and(one_mask_conf>self.thres,one_mask_pred!=y),dim=1)
		#certify = torch.all(torch.logical_and(one_mask_conf>self.thres,one_mask_pred==y),dim=1)

		pred = torch.ones_like(one_mask_conf[:,0])
		for i in range(len(pred)):
			pred_i = one_mask_pred[i][one_mask_conf[i]>self.thres]
			unique = torch.unique(pred_i)
			if len(unique) == 1:
				pred[i] = unique[0]
			else:
				pred[i] = -1
		return pred

	def certify(self,x,y):
		one_mask_conf,one_mask_pred = self.helper(x)
		y = y.unsqueeze(-1)
		certify = torch.all(torch.logical_and(one_mask_conf>self.thres,one_mask_pred==y),dim=1)
		return certify
class MRPC(MRPooling): # Minority Reports + PatchCURE
	def __init__(self,lrf,input_size,mask_size,mask_stride,thres=0):
		super().__init__(input_size,mask_size,mask_stride,thres)
		self.lrf = lrf #TODO, we could unify the API for SecureLayer and SecurePooling
	def _get_one_mask_logits(self,x,return_unmasked=True):
		unmasked_logits = self.lrf(x).unsqueeze(-1).unsqueeze(-1) if return_unmasked else None # [B,C,1,1] 
		one_masked_image = torch.einsum('bcij,nmcij -> bnmcij',x,self.mask_list) #[B,N1,N2,C,W,H]
		B,N1,N2,C,W,H = one_masked_image.shape
		one_masked_image = one_masked_image.flatten(end_dim=2) #[B*N1*N2,C,W,H]
		one_mask_logits = self.lrf(one_masked_image).view([B,N1,N2,-1]).permute(0,3,1,2)
		return one_mask_logits,unmasked_logits

