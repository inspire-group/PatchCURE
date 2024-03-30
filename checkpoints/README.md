## Checkpoints
### Download link: [link](https://drive.google.com/drive/folders/146Qy-FKgSKrzuaaSluafhm3jYDQzYERj?usp=sharing).

The entire folder could be large (~15GB). Pick what you need to download.

### checkpoint name format:

`{model_name}_{split_setting}_{training_technique}.pth.tar`

##### `{model_name}` options:

1. `vitsrf{wx}x{wy}` -- ViT-SRF with a window size of `wx`x`wy` -- available models `vitsrf14x2` `vitsrf14x1` `vitsrf2x2`
2. `bagnet{rf}` -- BagNet with a receptive field size of `rf` -- available models `bagnet17` `bagnet33` `bagnet45`
3. `resnet50`
4. `mae`
5. misc: `mael`  `vitlsrf14x2`  -- based on ViT-Large (others are based on ViT-Base)

##### `{split_setting}` options:

1. empty -- `k` is set to the largest value. Will use `utils.pcure.SecurePooling` to build a `PatchCURE` instance
2. `split{k}` -- k is the splitting layer index. Will use `utils.pcure.SecureLayer` to build a `PatchCURE` instance

##### `{training_technique}` options:

1. `masked` -- added random masks during the training (default setting used in the paper; always choose `masked` to reproduce results reported in the paper)
2. `vanilla` -- standard training



## Model training (optional)

Note: I didn't test the script and command discussed in this training section. Please feel free to reach out if run into any issue.

I did not include training scripts in this repository for simplicity

I used `timm`'s training script to train PatchCURE models. https://github.com/huggingface/pytorch-image-models/blob/main/train.py

#### Train a vanilla model (without adding masks for training)
```shell
# Train bagnet{rf}_vanilla (using the same hyper-parameters for different bagnet). 
# The hyper-parameters generally follow the ResNet-Strike-Back paper (Schedule B)
# PS: Train from scratch
./distributed_train.sh 8 imagenet --model bagnet33 --batch-size 64 --lr 0.0015 --epochs 300 --experiment bagnet33_64x8_0.0015 \
--opt lamb --sched cosine --warmup-epochs 5 --weight-decay 0.02 --drop-path 0.05 --aa rand-m7-mstd0.5-inc1 --cutmix 1.0 --mixup 0.1 --bce-loss --crop-pct 0.95 --amp

# Train vitsrf{wx}x{wy}_vanilla 
# The hyper-parameters generally follow the MAE paper 
# PS: Finetune MAE. mae_finetuned_vit_base.pth can be found in https://github.com/facebookresearch/mae/blob/main/FINETUNE.md
# PS: checkpoints/mae_vanilla.pth.tar is the same as mae_finetuned_vit_base.pth from MAE's repo (I might have changed the dict key from 'model' to 'state_dict')
./distributed_train.sh 8 data/imagenet     --opt adamw     --batch-size 64     --model vitsrf14x2     --initial-checkpoint checkpoints/mae_finetuned_vit_base.pth     --layer-decay 0.65     --weight-decay 0.05 --drop-path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25     --aa rand-m9-mstd0.5-inc1 --amp     --epochs 100  --lr-base 5e-5


```

#### Train a masked model (adding masks for training)

I finetuned vanilla models trained above for masked models.

```shell
# train bagnet{rf}_masked
./distributed_train.sh 8 data/imagenet --model bagnet33 --batch-size 64 --lr 0.0005 --epochs 50 --opt lamb --sched cosine --warmup-epochs 5 --weight-decay 0.02 --drop-path 0.05 --aa rand-m7-mstd0.5-inc1 --cutmix 1.0 --mixup 0.1 --bce-loss --crop-pct 0.95 --amp --initial-checkpoint checkpoints/bagnet33_vanilla.pth.tar

# train vitsrf{wx}x{wy}_masked
./distributed_train.sh 8 data/imagenet   --opt adamw  --batch-size 64  --model vitsrf14x2  --initial-checkpoint checkpoints/vitsrf14x2_vanilla.pth.tar   --layer-decay 0.65  --weight-decay 0.05 --drop-path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25  --aa rand-m9-mstd0.5-inc1 --amp  --epochs 50  --lr-base 2e-5

# train vitsrf{wx}x{wy}_split{k}_masked
./distributed_train.sh 8 data/imagenet   --model vitsrf14x2_split3  --opt adamw     --batch-size 64   \
     --layer-decay 0.65   --weight-decay 0.05 --drop-path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
 --aa rand-m9-mstd0.5-inc1 --amp --epochs 100 --lr-base 5e-5 --num-classes 1000  

```
See `build_model_training.py` for an ad-hoc script for building models and loading initial checkpoints for training. Replace the `model = create_model()` in https://github.com/huggingface/pytorch-image-models/blob/main/train.py with this loading function and then run `train.py`
The script is not tested. I will try to release another repo for training when I have time (no promise though).