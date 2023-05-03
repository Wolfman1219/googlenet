# [![Typing SVG](https://readme-typing-svg.herokuapp.com?font=Fira+Code&pause=1000&color=F7C5B1&background=89D1FF00&repeat=false&width=440&lines=GoogLeNet+with+TensorFlow+Keras)](https://git.io/typing-svg)
This repository contains the implementation of GoogLeNet architecture using TensorFlow Keras. GoogLeNet is a deep convolutional neural network (CNN) architecture that was introduced by Szegedy et al. in their 2014 paper, "Going Deeper with Convolutions." The architecture won the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) in 2014.

## GoogLeNet Architecture

GoogLeNet is known for its inception modules, which allow the network to learn complex features with fewer parameters compared to traditional CNNs. The architecture consists of 22 layers and uses 9 inception modules. The key components of GoogLeNet are:

1.Inception Module: The inception module is a building block of GoogLeNet that consists of multiple parallel convolutional layers with different filter sizes (1x1, 3x3, and 5x5) and a max-pooling layer. The outputs of these layers are concatenated to form the final output of the inception module. This design allows the network to learn features at different scales and reduces the number of parameters.

2.1x1 Convolution: GoogLeNet uses 1x1 convolutions to reduce the number of feature maps before applying more computationally expensive operations like 3x3 and 5x5 convolutions. This technique is known as "bottleneck layers" and helps to reduce the computational complexity of the network.

3.Global Average Pooling: Instead of using fully connected layers at the end of the network, GoogLeNet uses global average pooling to reduce the spatial dimensions of the feature maps to 1x1. This significantly reduces the number of parameters and prevents overfitting.

4.Auxiliary Classifiers: GoogLeNet introduces auxiliary classifiers connected to intermediate layers of the network. These classifiers provide additional supervision during training and help to improve the gradient flow in deeper layers.

### Usage
To usage first install requirement modules:
```shell
pip install -r requirements.txt
```
And run ```googlenet.py``` script:
```shell
python googlenet.py
```
You can modify ```googlenet.py``` for train own custom dataset.

For more information on how to use TensorFlow Keras for training and evaluation, please refer to the official TensorFlow Keras documentation.

## References

Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., ... & Rabinovich, A. (2015). Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1-9).
