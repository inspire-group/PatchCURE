import torch
from timm.models.vision_transformer import VisionTransformer


__all__ = ['vit_base_patch16_224_srf','vit_large_patch16_224_srf']  # model_registry will add each entrypoint fn to this


class VisionTransformer_SRF(VisionTransformer):
    def __init__(self, window_size, num_patch=(14,14),return_features=False,use_cls_token=True,**kwargs):
        super(VisionTransformer_SRF, self).__init__(**kwargs)
        self.num_patch = num_patch # number of patches in each axis
        self.window_size = window_size
        self.return_features = return_features # set it to True when we use the model for defense
        assert self.num_patch[0] % self.window_size[0] ==0
        assert self.num_patch[1] % self.window_size[1] ==0
        self.num_window = (self.num_patch[0] // self.window_size[0],self.num_patch[1] // self.window_size[1])
        assert self.global_pool == 'avg' # the setting of MAE. Our model follows the MAE architecture
        print('building a model with window_size',window_size)
        # all models used in the paper set use_cls_token=True. the original MAE architecture also use cls_token
        # Somehow I found adding cls_token make the model easier to train
        self.use_cls_token = use_cls_token 
        # some ad-hoc code for training with masks. can be ignored 
        tmp = {14:1,2:2,1:3}
        self.mask_size = (tmp[self.window_size[0]],tmp[self.window_size[1]])

    def forward(self, x):
        # based on the forward function in timm/models/vision_transformer.py
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        #x = self.patch_drop(x) #depend on the timm version. older version doesn't have this one
        x = self.norm_pre(x)

        B,N,C = x.shape # C is the embed_dim # [256, 197, 768]
        assert N == self.num_patch[0]*self.num_patch[1]+1
        x = x[:,1:] #[B,16x16,C]

        x = x.view(B, self.num_window[0], self.window_size[0], self.num_window[1], self.window_size[1], C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(-1, self.window_size[0]*self.window_size[1], C)
        
        if self.use_cls_token:
            cls_token = self.cls_token.expand(x.shape[0],-1,-1)
            x = torch.cat([cls_token,x],dim=1) #[B*16,16+1,C]
        
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)

        x = self.norm(x)

        if self.use_cls_token:
            x = x[:, self.num_prefix_tokens:]
        
        x = self.fc_norm(x)

        x = x.view(B, self.num_window[0], self.num_window[1], self.window_size[0], self.window_size[1], C) 
        x = self.head(x) # 
        if self.training: # ad hoc implementation for training with masks (can be ignored for now)
            x_idx = torch.randint(0 ,self.num_window[0]-self.mask_size[0]+1, (2,))
            y_idx = torch.randint(0 ,self.num_window[1]-self.mask_size[1]+1, (2,))
            mask = torch.ones(1,self.num_window[0],self.num_window[1],1,1,1,device=x.device)
            for xx,yy in zip(x_idx,y_idx):
                mask[0,xx:xx+self.mask_size[0],yy:yy+self.mask_size[1]]=0
            x = x*mask

        if self.return_features: 
            x = x.mean(dim=(3,4)).permute(0,3,1,2) # 
        else:
            x = x.mean(dim=(1,2,3,4))
        return x
    


def vit_base_patch16_224_srf(window_size=(14,2), return_features=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, global_pool='avg',window_size=window_size,return_features=return_features)
    model = VisionTransformer_SRF(**dict(model_kwargs, **kwargs))
    return model
def vit_large_patch16_224_srf(window_size=(14,2), return_features=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, global_pool='avg',window_size=window_size,return_features=return_features)
    model = VisionTransformer_SRF(**dict(model_kwargs, **kwargs))
    return model
