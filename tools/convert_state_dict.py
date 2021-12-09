import torch
import copy


# model_path="/cfs/cfs-31b43a0b8/personal/zilonghuang/models_and_config/topformerv1_md1_d4_large-224/topformerv1_md1_d4_large-224-75.3.pth"
# model_path="/cfs/cfs-31b43a0b8/personal/zilonghuang/models_and_config/topformerv1_md1_d4_middle-224/topformerv1_md1_d4_middle-224-72.3.pth"
# model_path="/cfs/cfs-31b43a0b8/personal/zilonghuang/models_and_config/topformerv1_md1_d4_small-224/topformerv1_md1_d4_small-224-66.2.pth"
# model_path="/cfs/cfs-31b43a0b8/personal/wqiangzhang/codebase/Topformer/ade20k_abstudy/topformer_md2_d4_simplehead_ade20k_base/iter_160000.pth"
model_path="/cfs/cfs-31b43a0b8/personal/wqiangzhang/codebase/Topformer/ade20k/topformer_md1_d4_small_simplehead_lr1p2en4-wd0p01_dp0p08/iter_160000.pth"

model_state = torch.load(model_path)
state_dict = model_state['state_dict']

new_dict = copy.deepcopy(model_state)
new_dict['state_dict'] = {}

for k, v in state_dict.items():
    if "patch_embed" in k:
        nk = k.replace("patch_embed", "tpm")
        new_dict['state_dict'][nk] = v
    elif "fuse_blocks" in k:
        nk = k.replace("fuse_blocks", "SIM")
        if "to_feat1" in nk:
            nk = nk.replace("to_feat1", "local_embedding")
        elif "to_feat2" in nk:
            nk = nk.replace("to_feat2", "global_embedding")
        elif "fuse2" in nk:
            nk = nk.replace("fuse2", "global_act")
        new_dict['state_dict'][nk] = v
    else:
        new_dict['state_dict'][k] = v
torch.save(new_dict, "/cfs/cfs-31b43a0b8/personal/wqiangzhang/work_dir/TopFormer/modelzoos/segmentation/topformer-T-512x512-ade20k-32.9.pth")