# Copyright (c) OpenMMLab. All rights reserved.

from mmseg.datasets.basesegdataset import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class SampleDataset(BaseSegDataset): 
    METAINFO = dict(
        classes=('display', 'background', 'black',
                'post', 'green', 'red') ,
        palette=[[27,  71 ,151],
                [111,  48 ,253],
                [254 ,233  , 3],
                [255  , 0,  29],
                [255 ,159 ,  0],
                [255 ,160  , 1]])
    
    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
