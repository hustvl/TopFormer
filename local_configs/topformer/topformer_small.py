# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)

cfgs_md2_middle = dict(
    cfg=[
    # k,  t,  c, s
        [3,   1,  16, 1], # 1/2        0.464K  17.461M
        [3,   4,  24, 2], # 1/4 1      3.44K   64.878M
        [3,   3,  24, 1], #            4.44K   41.772M
        [5,   3,  48, 2], # 1/8 3      6.776K  29.146M
        [5,   3,  48, 1], #            13.16K  30.952M
        [3,   3,  96, 2], # 1/16 5     16.12K  18.369M
        [3,   3,  96, 1], #            41.68K  24.508M
        [5,   6,  128, 2], # 1/32 7     0.129M  36.385M
        [5,   6,  128, 1], #            0.335M  49.298M
        [3,   6,  128, 1], #            0.335M  49.298M
    ],
    embed_out_indice=[2, 4, 6, 9],
    channels=[24, 48, 96, 128],
    decode_out_indices=[1, 2, 3],
    out_channels=[None, 192, 192, 192],
    num_heads=6,
    c2t_stride=2,
)

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='Topformer',
        cfgs=cfgs_md2_middle['cfg'], 
        channels=cfgs_md2_middle['channels'],
        out_channels=cfgs_md2_middle['out_channels'], 
        embed_out_indice=cfgs_md2_middle['embed_out_indice'],
        decode_out_indices=cfgs_md2_middle['decode_out_indices'],
        depths=4,
        num_heads=cfgs_md2_middle['num_heads'],
        c2t_stride=cfgs_md2_middle['c2t_stride'],
        drop_path_rate=0.1,
        norm_cfg=norm_cfg,
        init_cfg=dict(
            type='Pretrained', checkpoint='modelzoos/classification/topformer-S-224-72.3.pth')
    ),
    decode_head=dict(
        type='SimpleHead',
        in_channels=[192, 192, 192],
        in_index=[0, 1, 2],
        channels=192,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))