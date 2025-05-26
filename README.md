# T3Set: A Multimodal Dataset with Targeted Suggestions for LLM-based Virtual Coach in Table Tennis Training
This is the official repository for KDD'25 (dataset and benchmark track) paper.  
<p align="center">
<img width="800" src="./static/images/overview.png"/>
</p>

## Overview
T3Set (<b>T</b>able <b>T</b>ennis <b>T</b>raining) is a multimodal dataset with aligned video-sensor-text data in table tennis training.

The key features of T3Set include 
- temporal alignment between sensor data, video data, and text data. 
- high-quality targeted suggestions which are consistent with predefined suggestion taxonomy.

## Data Statistics
T3Set covers seven commonly used techniques in table tennis (i.e., attack, block, flick, pendulum, push, short, topspin). 
The whole dataset consists of 32 amateur players, 380 multi-ball training rounds, 8,655 strokes, and a total of 8,395 pieces of professional suggestions from coaches. 

## Dataset
Our dataset could be accessed through the Zenodo link:
XXX

*All participants signed informed consent forms and authorized open-source usage. The experimental procedure was approved by the laboratory's ethics review.*

Dataset License: CC BY-NC-ND 4.0

## Model and Script
We provide the model and script for the T3Set dataset. Please refer to corresponding folders.

## Citation
If you find our work useful, please consider citing our paper:
```
@inproceedings{
    ma2025t3set,
    title={T3Set: A Multimodal Dataset with Targeted Suggestions for LLM-based Virtual Coach in Table Tennis Training},
    author={Ji Ma and Jiale Wu and Haoyu Wang and Yanze Zhang and Xiao Xie and Zheng Zhou and Jiachen Wang and Yingcai Wu},
    year={2025},
    booktitle={Proceedings of the 31st SIGKDD Conference on Knowledge Discovery and Data Mining},
    pages={},
    doi={10.1145/3711896.3737407}
    }
```