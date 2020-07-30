# TC-ResNet implementation with Keras

This is a reimplementation of the TC-ResNet 8 and 14 architecture, proposed by Hyperconnect. The research aims for a lightweight Convolutional Neural Network model to solve the Keyword Spotting problem with audio data in real time on mobile devices.

[Original paper](https://arxiv.org/abs/1904.03814v2)

[Author's implementation with Tensorflow](https://github.com/hyperconnect/TC-ResNet)

# How to use

For training, testing and validation, this implementation uses [Google's Speech Command Dataset](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html). Please download the dataset, and extract into a folder named `dataset` in the root folder of the repository.

Run `main.py` to train the model.

Run `live.py` to demostrate live prediction of the model in real time.