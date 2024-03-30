import timm
import torch
from torch import nn


__all__ = ['split_resnet50_like','split_vit_like']  # model_registry will add each entrypoint fn to this

def split_resnet50_like(model,block_i):
	# split a resnet50/bagnet model, return two sub-models
	# resnet50: conv,bn,relu,maxpool,block1,block2,block3,block4,pool2d,fc
	# bagenet: conv,conv,bn,relu,block1,block2,block3,block4,avgpool,fc
	assert 0<block_i<=4 #4 blocks in resnet like architecture
	children = list(model.children())
	pre_layers = children[:4+block_i]
	post_layers = children[4+block_i:]
	return nn.Sequential(*pre_layers),nn.Sequential(*post_layers) # for ResNet, the model splitting is easy. we can use the sequential module  


# For ViT, we need to write a wrapper to specify the forward logic.. because it is more complex than nn.Sequential
class ViTPreModelWrapper(nn.Module): # wrapper: for first part of *vanilla* ViT 
	def __init__(self,model,pre_layers):
		super().__init__()
		self.pre_layers = nn.Sequential(*pre_layers)

		self.no_embed_class = model.no_embed_class
		self.pos_embed = model.pos_embed
		self.cls_token = model.cls_token
		self.pos_drop = model.pos_drop

	def _pos_embed(self, x): #https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L548
		if self.no_embed_class:
			# deit-3, updated JAX (big vision)
			# position embedding does not overlap with class token, add then concat
			x = x + self.pos_embed
			if self.cls_token is not None:
				x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
		else:
			# original timm, JAX, and deit vit impl
			# pos_embed has entry for class token, concat then add
			if self.cls_token is not None:
				x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
			x = x + self.pos_embed
		return self.pos_drop(x)

	def forward(self,x):
		x = self.pre_layers[0](x)
		x = self._pos_embed(x) # need to add this logic back
		for layer in self.pre_layers[1:]:
			x = layer(x)
		return x 

class ViTSRFPreModelWrapper(ViTPreModelWrapper):  # wrapper: for first part of *ViT-SRF*
	def __init__(self,model,pre_layers,use_cls_token=True):
		super().__init__(model,pre_layers)
		self.num_patch = model.num_patch
		self.num_window = model.num_window
		self.window_size = model.window_size
		self.cls_token = model.cls_token
        # all models used in the paper set use_cls_token=True. the original MAE architecture also use cls_token
		self.use_cls_token = use_cls_token

	def forward(self,x):
		x = self.pre_layers[0](x) #patch_embed
		x = self._pos_embed(x) # need to add this logic back
		x = self.pre_layers[1](x) #norm_pre
		B,N,C = x.shape # C is the embed_dim # [256, 197, 768]
		assert N == self.num_patch[0]*self.num_patch[1]+1
		x = x[:,1:] #[B,num_patch[0]xnum_patch[1],C]

		x = x.view(B, self.num_window[0], self.window_size[0], self.num_window[1], self.window_size[1], C)
		x = x.permute(0, 1, 3, 2, 4, 5).contiguous()#.view(-1, self.window_size[0], self.window_size[1], C)#########################
		x = x.view(-1, self.window_size[0]*self.window_size[1], C)
		self.use_cls_token = True
		if self.use_cls_token:
			cls_token = self.cls_token.expand(x.shape[0],-1,-1)
			x = torch.cat([cls_token,x],dim=1) #[B*num_window[0]*num_window[1],1+window_size[0]*window_size[1],C]

		for layer in self.pre_layers[2:]:
			x = layer(x)


		# additional operations to make the output shape compatible for deeper vanilla vit layers

		if self.use_cls_token:
			x = x.view(B, self.num_window[0], self.num_window[1], (1+self.window_size[0]*self.window_size[1])*C)
		else:
			x = x.view(B, self.num_window[0], self.num_window[1], (self.window_size[0]*self.window_size[1])*C)

		x = x.permute(0,3,1,2)

		return x 

class ViTPostModelWrapper(nn.Module):  # wrapper: for last part of ViT and ViT-SRF
	def __init__(self,model,post_layers,use_cls_token=True):
		super().__init__()

		self.global_pool = model.global_pool
		self.num_prefix_tokens = model.num_prefix_tokens
		self.post_layers = nn.Sequential(*post_layers)

        # all models used in the paper set use_cls_token=True. the original MAE architecture also use cls_token
		self.use_cls_token = use_cls_token
		self.tmp = {14:1,2:2,1:3} # adhoc implementation for training; can be ignored

	def forward(self,x):
		self.use_cls_token = True
		if len(x.shape)==4: # input from vitsrf # input from vanilla vit should be B,N,C three dimensions
			x = x.permute(0,2,3,1)
			B = x.shape[0]
			if self.use_cls_token:
				x = x.view(B, self.num_window[0], self.num_window[1], 1+self.window_size[0]*self.window_size[1],-1)
			else:
				x = x.view(B, self.num_window[0], self.num_window[1], self.window_size[0]*self.window_size[1],-1)

			C = x.shape[-1]
			if self.training: # adhoc implementation for training with masks
				self.mask_size = (self.tmp[self.window_size[0]],self.tmp[self.window_size[1]])
				x_idx = torch.randint(0 ,self.num_window[0]-self.mask_size[0]+1, (2,))
				y_idx = torch.randint(0 ,self.num_window[1]-self.mask_size[1]+1, (2,))
				mask = torch.ones(1,self.num_window[0],self.num_window[1],1,1,device=x.device)
				for xx,yy in zip(x_idx,y_idx):
					mask[0,xx:xx+self.mask_size[0],yy:yy+self.mask_size[1]]=0
				x = x*mask

			if self.use_cls_token: # I calculate mean for nonzero cls token as the cls token for the LRF sub-model
				cls_token = x[:,:,:,0]
				tmp = torch.sum(cls_token,dim=-1) == 0 
				cls_token[tmp] = torch.nan 
				cls_token = torch.nanmean(cls_token,dim=(1,2)).view(B,1,C)
				x = x[:,:,:,1:].view(B, self.num_window[0], self.num_window[1], self.window_size[0], self.window_size[1],-1)
				x = x.permute(0,1,3,2,4,5).contiguous().view(B,self.num_patch[0]*self.num_patch[1],C) # 
				x = torch.cat((cls_token, x), dim=1)
			else:
				x = x.view(B, self.num_window[0], self.num_window[1], self.window_size[0], self.window_size[1],-1)
				x = x.permute(0,1,3,2,4,5).contiguous().view(B,self.num_patch[0]*self.num_patch[1],C) # 

		for layer in self.post_layers[:-2]:
			x = layer(x)

		if self.global_pool:
			if self.use_cls_token:
				x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
			else:
				x = x.mean(dim=1) 
		x = self.post_layers[-2](x)
		x = self.post_layers[-1](x)

		return x



def split_vit_like(model,block_i,srf=False): 
	# split a ViT-Base model, return two sub-models
	children = list(model.children())
	assert 0<=block_i<=12 # attention layers
	for i,m in enumerate(children): #timm.vision_transformer's attention layers are wrapped within a sequential module. need to find this one first
		if isinstance(m,nn.Sequential):
			break
	pre_attention_layers = children[:i]
	attention_blocks = children[i] #attention_blocks is a nn.Sequential module with all attention layers # this index number might vary, according to the implementation and init parameters of vit
	post_attention_layers = children[i+1:]
	attention_children = list(attention_blocks.children())
	pre_layers = pre_attention_layers + attention_children[:block_i]
	post_layers = attention_children[block_i:] + post_attention_layers
	pre_model = ViTSRFPreModelWrapper(model,pre_layers) if srf else ViTPreModelWrapper(model,pre_layers)
	post_model = ViTPostModelWrapper(model,post_layers)
	return pre_model,post_model

