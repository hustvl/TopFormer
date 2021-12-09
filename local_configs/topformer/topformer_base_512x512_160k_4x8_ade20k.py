_base_ = [
    './topformer_base_512x512_160k_2x8_ade20k.py'
]

data=dict(samples_per_gpu=4)
