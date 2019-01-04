# Learning-bit-codes-for-images

The repository contains an implementation for Image2Vec.
It uses a pretrained AlexNet with a latent hashing layer to learn bit vectors.

The code is based on this [publication](http://www.iis.sinica.edu.tw/~kevinlin311.tw/cvprw15.pdf) made in [CVPR](http://cvpr2019.thecvf.com/) in 2015.

## Requirements

- Python 3
- TensorFlow >= 1.8rc0
- Numpy

## Usage

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
