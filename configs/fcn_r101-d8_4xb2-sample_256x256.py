_base_ = ['_base_/models//fcn_r50-d8.py', ] 

#CONFIG==============  
work_dir = 'projects/wandb_sweep/exp'
runner_type = 'SweepRunner'

custom_imports = dict(
    imports=['projects.wandb_sweep.mmseg.datasets.sample_dataset',
             'projects.wandb_sweep.mmengine.runner.runner',  
             'projects.wandb_sweep.mmengine.hooks.wandb_sweep_hook'
             ], 
)   
custom_hooks = [dict(
    type='WandbSweepHook'
) ]

experiment_name = 'sweep-test'
wandb_sweep = dict(
    cfg ='/mmsegmentation/projects/wandb_sweep/configs_sweep/config.yaml', 
    project_name = 'wandb-sweep-test' 
)

#MODEL============== 
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size= (256,256),
    mean=None,
    std=None,
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    data_preprocessor = data_preprocessor, 
    decode_head = dict(
        num_classes = 6,
    )
)
#DATASET===================
# dataset settings
dataset_type = 'SampleDataset'
data_root = 'data/sample/'
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'), 
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', direction=['horizontal', 'vertical'], prob=[0.25,0.25]),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]  
train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/train', seg_map_path='annotations/train'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/test', seg_map_path='annotations/test'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator


#SCHEDULES===================
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005) 
optim_wrapper = dict(type='OptimWrapper', 
                     optimizer=optimizer,
                      clip_grad=None) 
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=20000,
        by_epoch=False)
]
# training schedule for 100
train_cfg = dict(type='IterBasedTrainLoop', max_iters=100, val_interval=100)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=-1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))


#DEFAULT RUNTIME===================
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend'), 
                dict(type = 'WandbVisBackend',)] 
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

tta_model = dict(type='SegTTAModel')
