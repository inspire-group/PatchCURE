# undefended models (Table 3)
python main.py --model  mae_masked --undefended # --runtime --flops --memory
python main.py --model  resnet50 --undefended # --runtime --flops --memory
python main.py --model  bagnet33_masked --undefended  # --runtime --flops --memory
python main.py --model  vitsrf14x2_masked --undefended # --runtime --flops --memory
python main.py --model  vitsrf14x1_masked --undefended # --runtime --flops --memory
python main.py --model  vitsrf2x2_masked --undefended # --runtime --flops --memory


# PatchCURE (Table 4)
### SRF-only PatchCURE (split k=12 for ViT; k=50 for BagNet)
python main.py --model  vitsrf14x2_masked --patch-size 32 --mask-stride 1 --certify  # --runtime --flops --memory
python main.py --model  vitsrf14x1_masked --patch-size 32 --mask-stride 1 --certify  # --runtime --flops --memory
python main.py --model  vitsrf2x2_masked --patch-size 32 --mask-stride 1 --certify  # --runtime --flops --memory
python main.py --model  bagnet45_masked --patch-size 32 --mask-stride 1 --certify  # --runtime --flops --memory
python main.py --model  bagnet33_masked --patch-size 32 --mask-stride 1 --certify  # --runtime --flops --memory
python main.py --model  bagnet17_masked --patch-size 32 --mask-stride 1 --certify  # --runtime --flops --memory
### SRF+LRF PatchCURE
python main.py --model  vitsrf14x2_split11_masked --patch-size 32 --mask-stride 1 --certify  # --runtime --flops --memory
python main.py --model  vitsrf14x2_split10_masked --patch-size 32 --mask-stride 1 --certify  # --runtime --flops --memory
python main.py --model  vitsrf14x2_split9_masked --patch-size 32 --mask-stride 1 --certify  # --runtime --flops --memory
python main.py --model  vitsrf14x2_split6_masked --patch-size 32 --mask-stride 1 --certify  # --runtime --flops --memory
python main.py --model  vitsrf14x2_split3_masked --patch-size 32 --mask-stride 1 --certify  # --runtime --flops --memory
python main.py --model  vitsrf14x2_split2_masked --patch-size 32 --mask-stride 1 --certify  # --runtime --flops --memory
python main.py --model  vitsrf14x2_split1_masked --patch-size 32 --mask-stride 1 --certify  # --runtime --flops --memory
### LRF-only PatchCURE (k=0)
python main.py --model  vitsrf14x2_split0_masked --patch-size 32 --mask-stride 1 --certify  # --runtime --flops --memory
python main.py --model  vitsrf14x1_split0_masked --patch-size 32 --mask-stride 1 --certify  # --runtime --flops --memory
python main.py --model  vitsrf2x2_split0_masked --patch-size 32 --mask-stride 1 --certify  # --runtime --flops --memory
### more model checkpoints are available in checkpoints/README.md

# PatchCURE with different patch sizes
python main.py --model  vitsrf14x2_masked --patch-size 48 --mask-stride 1 --certify  
python main.py --model  vitsrf14x2_masked --patch-size 64 --mask-stride 1 --certify  
python main.py --model  vitsrf14x2_masked --patch-size 80 --mask-stride 1 --certify  
python main.py --model  vitsrf14x2_masked --patch-size 96 --mask-stride 1 --certify  
# .....


# PatchCleanser #{6:33,5:39,4:49,3:65,2:97}
python main.py --model  mae_masked --patch-size 32 --mask-stride 33 --certify # num_mask 6x6  
python main.py --model  mae_masked --patch-size 32 --mask-stride 39 --certify # num_mask 5x5  
python main.py --model  mae_masked --patch-size 32 --mask-stride 49 --certify # num_mask 4x4  
python main.py --model  mae_masked --patch-size 32 --mask-stride 65 --certify # num_mask 3x3
python main.py --model  mae_masked --patch-size 32 --mask-stride 97 --certify # num_mask 2x2
