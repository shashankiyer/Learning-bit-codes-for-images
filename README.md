# Learning-bit-codes-for-images

The repository contains an implementation for Image2Vec and Similarity Search.
It uses a pretrained AlexNet with a latent hashing layer to learn bit vectors.

The code is based on this [publication](http://www.iis.sinica.edu.tw/~kevinlin311.tw/cvprw15.pdf) made at CVPR in 2015.

Here is a link to my [blog](https://medium.com/@shashank_iyer/learning-bit-codes-for-images-e09966891acc) for a better understanding of how this process works.

## Requirements

- Python 3
- TensorFlow >= 1.8rc0
- Numpy

## Setup

To replicate the experiment I performed, you will need to download the CIFAR10 image dataset.

Create a directory called 'data' in the root of this repository.

Download and unzip the [CIFAR10 dataset](https://github.com/thulab/DeepHash/releases/download/v0.1/cifar10.zip) in the data folder.

Make sure the tree of ```/path/to/project/data/cifar10``` looks like:
```
.
|-- database.txt
|-- test
|-- test.txt
|-- train
`-- train.txt
```
In addition, you will also need to download the pretrained [AlexNet weights](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy). Ensure they are stored as:
```
data/pretrained_alexnet/bvlc_alexnet.npy
```

If you would like to use a custom database, you have to create three `.txt` files (`train.txt`, `test.txt` and `database.txt`). Each of them list the complete path to your train/test/database images together with the class number as a 1-hot vector in the following structure.

```
Example train.txt:
/path/to/train/image1.png 1 0 0 0 0 0 0 0 0 0
/path/to/train/image2.png 0 1 0 0 0 0 0 0 0 0
/path/to/train/image3.png 0 0 1 0 0 0 0 0 0 0
/path/to/train/image4.png 1 0 0 0 0 0 0 0 0 0
.
.
```
Create a sub-directory within data to store these text files as:
```
data/my_dataset/

```
You can tweak training parameters by modifying ```params.json``` in ```experiments\```

## Tensorboard
The code saves summaries so you can track the training and validation.
```
tensorboard --logdir experiments/checkpoint_data
```

## Getting things to run

1. Get access to a system with a recent version of tensorflow and a GPU.
   We used the Ubuntu Deep Learning image (v=20) and a g3s.xlarge available on AWS EC2. 

   Note, once you login you will need to install Tensorflow by running ```conda install tensorflow-gpu```

2. Clone the repo and follow the **Setup** instructions.

3. You may finetune the model by running ```python finetune.py```.
   This will display information to indicate finetuning is in progress.

4. Let it run for a while (I recommend 20k steps).
   You can watch this process by launnching Tensorboard.
   
   A. If you are using an Amazon EC2 instance, you will need to create a "tunnel" by running:
   
   ```ssh -L 127.0.0.1:6006:127.0.0.1:6006 -i path/to/your/key ubuntu@public_dns```
   
   B. Launch Tensorboard by running:
   
   ```tensorboard --logdir path/to/project/experiments/checkpoint_data```
   
   C. On your local machine, you may browse to ```localhost:6006``` to view the results.

5. You may stop finetuning with Ctrl-C. This will give you an opportunity to get some precision numbers.

   Run ```python similarity_search.py```
   
   This will report mAP ( mean average precision ) that is well described [here](https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173).

6. You may resume finetuning by running ```python finetune.py```
