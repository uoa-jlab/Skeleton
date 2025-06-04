# Skeleton Predict

Collect plantar pressure data using smart insoles and then perform prediction using a Transformer model.

---

## Table of Contents

- [Introduction](#Introduction)
- [Install](#Intall)
- [Project-Structure](#Project-Structure)
- [项目结构](#项目结构)
- [数据说明](#数据说明)
- [训练与测试](#训练与测试)
- [结果展示](#结果展示)
- [依赖项](#依赖项)
- [参考文献](#参考文献)
- [贡献](#贡献)
- [许可证](#许可证)

---

## Introduction

The primary function of this program is to reconstruct the skeletal data captured by OptiTrack using pressure data from both feet. Since the initial version was developed using Jupyter Notebook, you can directly download the Code.ipynb file and run it after installing the libraries listed in requirements.txt. If you prefer to use other IDEs such as PyCharm or VS Code, you will need to download the additional .py files. Please note that these files have not been thoroughly tested, so there might be some minor issues.

Since this project is highly task-specific, it cannot be directly applied to other programs. However, the overall structure is quite simple, and once the underlying idea is understood, it can be adapted to other applications through modifications.

---

## Install

All the required libraries are listed in the requirements.txt file.
```bash
pip install -r requirements.txt

```

---

## Project-Structure

```bash
Skeleton/
├── Code.ipynb/ # Complete code written in Jupyter
├── data_load.py/ # Loading the collected pressure and skeleton data
├── loss.py/ # Includes the loss function used for training and the loss function used to compute the deviation of the predicted results.
├── main.py/ # The main function of the program, consisting of model training.
├── model.py # Transformer model
├── predict.py # The program that performs prediction using the model
├── requirements.txt # The libraries required to run the program
├── util.py/ # Includes all the utilities or functions required to run the program
├── visualization # Visualize and compare the predicted skeleton data with the ground truth skeleton data
└── README.md # README
```
## Results Demo

Click the image below to watch the demo video:

[![Watch the demo](https://img.youtube.com/vi/dQw4w9WgXcQ/hqdefault.jpg)](https://www.youtube.com/watch?v=dQw4w9WgXcQ)
