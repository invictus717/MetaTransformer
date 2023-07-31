export CUDA_VISIBLE_DEVICES=0

# Scaobjectnn 
# pointnet++ 
# Loading ckpt @E241, val_macc 0.7793, val_oa 0.8033
# acc per cls is: [0.57831323 0.8743719  0.37593985 0.8172043  0.9230769  0.6666667
#  0.8480392  0.8952381  0.8215768  0.6703704  0.7818182  0.8095238
#  0.78333336 0.9380952  0.90588236]
# ckpt_pth=pretrain/scanobjectnn/scanobjectnn-train-pointnet++-ngpus1-20220331-210646-WswbXmuKRKWtk7mWLhJGTX/checkpoint/scanobjectnn-train-pointnet++-ngpus1-20220331-210646-WswbXmuKRKWtk7mWLhJGTX_ckpt_best.pth
# bash script/main_classification.sh cfgs/scanobjectnn/pointnet++.yaml wandb.use_wandb=False mode=test pretrained_path=$ckpt_pth 


# pointnet++para
# Loading ckpt @E184, val_macc 0.8322, val_oa 0.8584
# acc per cls is: [0.5903614  0.85427135 0.6766917  0.8736559  0.97179484 0.7266667
#  0.88235295 0.94285715 0.8589212  0.7777778  0.8454546  0.82857144
#  0.84166664 0.97619045 0.8352941 ]
# ckpt_pth=pretrain/scanobjectnn/scanobjectnn-train-pointnet++para-ngpus1-20220331-225749-A2A7QdqG3dKQ2WzjgsqsSZ/checkpoint/scanobjectnn-train-pointnet++para-ngpus1-20220331-225749-A2A7QdqG3dKQ2WzjgsqsSZ_ckpt_best.pth
# CUDA_VISIBLE_DEVICES=0 bash script/main_classification.sh cfgs/scanobjectnn/pointnet++para.yaml wandb.use_wandb=False mode=test pretrained_path=$ckpt_pth 

# pointnext-s
# Loading ckpt @E233, val_macc 0.8648, val_oa 0.8810
# acc per cls is: [0.7108434  0.8844221  0.7368421  0.8844086  0.96410257 0.82666665
#  0.89705884 0.96666664 0.90041494 0.7888889  0.8363636  0.8857143
#  0.825      0.947619   0.91764706]
# ckpt_pth=pretrain/scanobjectnn/scanobjectnn-train-pointnext-s-ngpus1-20220331-210738-LZsAHnzDUmMPAGHJ86pX46/checkpoint/scanobjectnn-train-pointnext-s-ngpus1-20220331-210738-LZsAHnzDUmMPAGHJ86pX46_ckpt_best.pth
# CUDA_VISIBLE_DEVICES=0 bash script/main_classification.sh cfgs/scanobjectnn/pointnext-s.yaml wandb.use_wandb=False mode=test pretrained_path=$ckpt_pth 