A pytorch implementation of the paper 
[Detect to Track and Track to Detect](https://arxiv.org/abs/1710.03958).

## Introduction

This project is a pytorch implementation of 
detect to track and track to detect. 
This repository is influenced by the following implementations:

* [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch), based on Pytorch

* [rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn), based on Pycaffe + Numpy

* [longcw/faster_rcnn_pytorch](https://github.com/longcw/faster_rcnn_pytorch), based on Pytorch + Numpy

* [endernewton/tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn), based on TensorFlow + Numpy

* [ruotianluo/pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn), Pytorch + TensorFlow + Numpy

During our implementation, we refer to the above implementations,
especially 
[jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch). 
As in that implementation, this repository has the following qualities: 

* **It is pure Pytorch code**. We convert all the numpy implementations to pytorch!

* **It supports multi-image batch training**. We revise all the layers, including dataloader, rpn, roi-pooling, etc., 
to support multiple images in each minibatch.

* **It supports multiple GPUs training**. We use a multiple GPU wrapper (nn.DataParallel here) to make it flexible to 
use one or more GPUs, as a merit of the above two features.

Furthermore, since the Detect to Track and Track to Detect implementation 
originally used an R-FCN siamese network and correlation layer, we've added/modified the following:
* **Supports multiple images per roidb entry**. By default, we use 2 images in contiguous frames to define 
an roidb entry to faciliate
a forward pass through a two-legged siamese network. 

* **It is memory efficient**. We limit the aspect ratio of the images in each roidb and group images 
with similar aspect ratios into a minibatch. As such, we can train resnet101 with batchsize = 2 (4 images) on a 2 Titan X (12 GB). 

* **Supports 4 pooling methods**. roi pooling, roi alignment, roi cropping, and position-sensitive roi pooling. 
More importantly, we modify all of them to support multi-image batch training.

* **Supports correlation layer**. We adopt the correlation layer from NVIDIA's 
[flownet2](https://github.com/NVIDIA/flownet2-pytorch) implementation.

## Other Resources

* Original Matlab implementation by the authors [feichtenhofer/Detect-Track](https://github.com/feichtenhofer/Detect-Track)

## Benchmarking
*WORK IN PROGRESS*

This project is a work in progress, and PRs are welcome. The current implementation is
benchmarked against the Imagenet VID dataset.

For training, we adopt the common heuristic of passing alternating samples
from VID and DET (e.g. iteration 1 is from VID, iteration 2 is from DET, etc).
Additionally, for training, 10 frames are sampled per video snippet. This
avoids biasing the training towards longer snippets. However, validation performance
is evaluated on each frame from each snippet of VAL. Please refer to the
D&T paper for more details.

1). Baseline single-frame RFCN (see [this](https://github.com/Feynman27/pytorch-detect-rfcn) repo:
(Trained model can be accessed 
[here](https://drive.google.com/drive/folders/1TM9bJ1mod2EipgXHhYscRxkJhrtOGSju?usp=sharing) 
under the name rfcn_detect.pth

Imagenet VID+DET (Train/Test: imagenet_vid_train+imagenet_det_train/imagenet_vid_val,
scale=600, PS ROI Pooling).

model    | #GPUs | batch size | lr        | lr_decay | max_epoch     |  time/epoch | mem/GPU | mAP
---------|--------|-----|--------|-----|-----|-------|--------|-----
Res-101     | 2 | 2 | 1e-3 | 5   | 11   |  -- | 8021MiB   | 70.3

2). D(&T loss) Imagenet VID+DET
(Train/Test: imagenet_vid_train+imagenet_det_train/imagenet_vid_val, scale=600, PS ROI Pooling).
This network is initialized with the weights from the single-frame RFCN baseline above.
Trained model can be accessed from 
[here](https://drive.google.com/drive/folders/1TM9bJ1mod2EipgXHhYscRxkJhrtOGSju?usp=sharing) 
under the name rfcn_detect_track_1_7_32941.pth).

*Currently, the performance drops by 1.6 percentage points*. 
The issue is currently unknown. Again, PRs are welcome.


model    | #GPUs | batch size | lr        | lr_decay | max_epoch     |  time/epoch | mem/GPU | mAP 
---------|--------|-----|--------|-----|-----|-------|--------|-----
Res-101     | 2 | 2 | 1e-4 | 5   | 7   |  -- | 8021MiB   | 68.7

TODO: Result using Viterbi algorithm as linking post-processing step.

* If not mentioned, the GPU we used is NVIDIA Titan X Pascal (12GB).

### prerequisites

* Python 2.7
* Pytorch 0.3.0 (0.4.0+ may work, but hasn't been tested; 
some minor tweaks are probably required.)
* CUDA 8.0 or higher

TODO:
- Update to Pytorch 0.4.0+
- Make Python 3 compatible

### Build 
As pointed out by [ruotianluo/pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn), choose the right `-arch` to compile the cuda code:

  | GPU model  | Architecture |
  | ------------- | ------------- |
  | TitanX (Maxwell/Pascal) | sm_52 |
  | GTX 960M | sm_50 |
  | GTX 1080 (Ti) | sm_61 |
  | Grid K520 (AWS g2.2xlarge) | sm_30 |
  | Tesla K80 (AWS p2.xlarge) | sm_37 |
  
More details about setting the architecture can be found [here](https://developer.nvidia.com/cuda-gpus) or [here](http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

Install all the python dependencies using pip:
```
pip install -r requirements.txt
```

If you would like to use tensorboard, install the cpu version of
Tensorflow and install [TensorboardX](https://github.com/lanpa/tensorboard-pytorch)

Compile the cuda dependencies using following simple commands:

```
cd lib
sh make.sh
```

It will compile all the modules you need, including NMS, PSROI_POOLING, ROI_Pooing, ROI_Align and ROI_Crop. 
The default version is compiled with Python 2.7, please compile by yourself if you are using a different python version.

**As pointed out in this [issue](https://github.com/jwyang/faster-rcnn.pytorch/issues/16), if you encounter some error during the compilation, you might miss to export the CUDA paths to your environment.**

## Training


Then:
```
cd pytorch-detect-and-track
mkdir data
```

Download the ILSVRC VID and DET (train/val/test lists can be found [here](https://drive.google.com/drive/u/0/folders/1hFcVKwqrMFnXf7ysl8ENE6uf2hD7w9Zo). 
The ILSVRC2015 images can be downloaded from [here](http://image-net.org/download-images)
).

Untar the file:
```bash
tar xf ILSVRC2015.tar.gz
```
We'll refer to this directory as `$DATAPATH`.
Make sure the directory structure looks something like:
```bash
|--ILSVRC2015
|----Annotations
|------DET
|--------train
|--------val
|------VID
|--------train
|--------val
|----Data
|------DET
|--------train
|--------val
|------VID
|--------train
|--------val
|----ImageSets
|------DET
|------VID
```

Create a soft link under `pytorch-detect-and-track/data`:

```bash
ln -s $DATAPATH/ILSVRC2015 ./ILSVRC
```

Create a directory called `pytorch-detect-and-track/data/pretrained_model`,
and place the pretrained models into this directory.

Before training, set the correct directory to save and load the trained models.
The default is `./output/models`.
Change the arguments "save_dir" and "load_dir" in trainval_net.py and test_net.py to adapt to your environment.

To train an RFCN D&T model with resnet-101 on Imagenet VID+DET, simply run:
```
CUDA_VISIBLE_DEVICES=0,1 python trainval_net.py \
    --cuda \
    --mGPUs \
    --nw 12 \
    --dataset imagenet_vid+imagenet_det \
    --cag \
    --lr 1e-4 \
    --bs 2 \
    --lr_decay_gamma=0.1 \
    --lr_decay_step 3 \
    --epochs 10 \
    --use_tfboard True
```
where 'bs' is the batch size, `--cag` is a flag for class-agnostic bbox regression,
`lr`, `lr_decay_gamma`, and `lr_decay_step` are the learning rate, factor to decrease the
learning rate by, and the number of epochs before decaying the learning rate, respectively.
Above, `--bs`, `--nw` (number of workers; check with linux `nproc`),
and `--mGPUs` should be set according to the number of
GPUs you wish to train on and your GPU memory size.
**On 2 Titan Xps with 12G memory, the batch size can be up to 2 (4 images, 2 per GPU)**.


## Authorship

Contributions to this project have been made by [Thomas Balestri](https://github.com/Feynman27) and 
[Jugal Sheth](https://github.com//jugalsheth92).

