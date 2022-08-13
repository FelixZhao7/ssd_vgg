# ssd_vgg
### Table of Contents
- <a href='#installation'>Installation</a>
- <a href='#datasets'>Datasets</a>
- <a href='#training-ssd'>Train</a>
- <a href='#evaluation'>Evaluate</a>


&nbsp;
&nbsp;
&nbsp;
&nbsp;

## Installation
- Install [PyTorch](http://pytorch.org/) by selecting your environment on the website and running the appropriate command.
- Clone this repository.
- Then download the dataset by following the [instructions](#datasets) below.

## Datasets
To make things easy,bash scripts are provided to handle the dataset downloads and setup.


### COCO
Microsoft COCO: Common Objects in Context

##### Download COCO 2014
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/COCO2014.sh
```

### VOC Dataset
PASCAL VOC: Visual Object Classes

##### Download VOC2007 trainval & test
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
```

##### Download VOC2012 trainval
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2012.sh # <directory>
```

## Training SSD
- First download the fc-reduced VGG-16 PyTorch base network weights at:              https://1drv.ms/u/s!Asxoawb369ilchFdNBSY34T2O-8?e=zapfne
- Download the file in the `ssd_vgg/weights` dir



- To train SSD using the train script simply specify the parameters listed in `train.py` as a flag or manually change them.

```Shell
python train.py
```

- Note:
  * For training, an NVIDIA GPU is strongly recommended for speed.


## Evaluation
- If you don't want to spend time training, you can download the already trained SSD model weights at: https://1drv.ms/u/s!Asxoawb369ilcyI29ztq-vfpyAg?e=5g617p
- Download the file in the `ssd_vgg/weights` dir
- To evaluate a trained network:

```Shell
python eval.py
```

You can specify the parameters listed in the `eval.py` file by flagging them or manually changing them.  



