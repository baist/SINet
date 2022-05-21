## A codebase for video-based person re-identification

Salient-to-Broad Transition for Video Person Re-identiﬁcation (CVPR 2022)

SANet: Statistic Attention Network for Video-Based Person Re-Identiﬁcation (TCSVT 2021)

## Get started

```Shell
  # Train
  python main.py \
   --arch ${sinet, sbnet, idnet, sanet} \
   --dataset ${mars, lsvid, ...} \
   --root ${path of dataset} \
   --gpu_devices 0,1 \
   --save_dir ${path for saving modles and logs} \
  
  # Test with all frames
  python main.py \
   --arch ${sinet, sbnet, idnet, sanet} \
   --dataset mars \
   --root ${path of dataset} \
   --gpu_devices 0,1 \
   --save_dir ${path for saving logs} \
   --evaluate --all_frames --resume ${path of pretrained model}
  ```
  

## Pretrained Model

### MARS
|     Methods                   | Paper | Reproduce | Download |
|----- | -----| ----- | -----| 
| **SBNet** (ResNet50 + SBM)        | 85.7/90.2 | 85.6/90.7 | -- | 
| IDNet (Resnet50 + IDM)        | 85.9/90.5 | 85.9/90.4 | -- |
| SINet (ResNet50 + SBM + IDM)  | 86.2/91.0 | 86.3/90.9 | -- | 
|----- | -----| ----- | -----| 
| **SANet** (ResNet50 + SA Block) | 86.0/91.2 | 86.7/91.2 | -- | 


### LS-VID

|     Methods                   | Paper | Reproduce | Download |
|----- | -----| ----- | -----| 
| **SBNet** (ResNet50 + SBM)        | 77.1/85.1 | 77.2/85.3 | -- | 
| IDNet (Resnet50 + IDM)        | 78.0/86.2 | 78.2/86.0 | -- |
| SINet (ResNet50 + SBM + IDM)  | 79.6/87.4 | 79.9/87.2 | -- | 

### Citation

If you use our code in your research or wish to refer to the baseline results, please use the following BibTeX entry.
@inproceedings{bai2022SInet,
    title={Salient-to-Broad Transition for Video Person Re-identiﬁcation},
    author={Bai, Shutao and Ma, Bingpeng and Chang, Hong and Huang, Rui and Chen, Xilin},
    booktitle={CVPR},
    year={2022},
}

@ARTICLE{9570321,
  author={Bai, Shutao and Ma, Bingpeng and Chang, Hong and Huang, Rui and Shan, Shiguang and Chen, Xilin},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={SANet: Statistic Attention Network for Video-Based Person Re-Identification}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2021.3119983}}


## Acknowledgments

This code is based on the implementations of [**AP3D**](https://github.com/guxinqian/AP3D).
