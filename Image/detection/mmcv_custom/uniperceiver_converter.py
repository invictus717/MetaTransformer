import torch

checkpoint = torch.load("../pretrained/uni-perceiver-large-L24-H1024-224size-pretrained.pth",
                        map_location=torch.device('cpu'))
checkpoint = checkpoint['model']
new_checkpoint = {}
for k, v in checkpoint.items():
    new_k = k.replace("fused_encoder.", "")
    new_k = new_k.replace("in_proj_", "in_proj.")
    new_k = new_k.replace("video_embed.", "visual_embed.")
    new_k = new_k.replace("visual_embed.embeddings.weight",
                          "visual_embed.patch_embed.proj.weight")
    new_k = new_k.replace("visual_embed.embeddings.bias",
                          "visual_embed.patch_embed.proj.bias")
    new_k = new_k.replace("visual_embed.embeddings_st_pos.spatial_pos_embed.weight",
                          "visual_embed.patch_embed.spatial_pos_embed.weight")
    new_k = new_k.replace("visual_embed.embeddings_st_pos.temporal_pos_embed.weight",
                          "visual_embed.patch_embed.temporal_pos_embed.weight")

    if "loss_prepare" in new_k:
        pass
    elif "token_embed" in new_k:
        pass
    else:
        new_checkpoint[new_k] = v
        
for k, v in new_checkpoint.items():
    print(k, v.shape)

torch.save(new_checkpoint,
           "../pretrained/uni-perceiver-large-L24-H1024-224size-pretrained_converted.pth")
print("saved!")

