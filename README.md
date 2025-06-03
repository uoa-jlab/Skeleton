# Skeleton Predict

Collect plantar pressure data using smart insoles and then perform prediction using a Transformer model.

---

## Table of Contents

- [Introduction](#Introduction)
- [Install](#Intall)
- [使用方法](#使用方法)
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

## Project Structure

project-name/
├── data/ # Dataset files
├── docs/ # Documentation files
├── notebooks/ # Jupyter notebooks for experiments and analysis
├── src/ # Source code
│ ├── init.py # Package initialization
│ ├── model.py # Model definitions
│ ├── train.py # Training scripts
│ └── utils.py # Utility functions
├── tests/ # Unit tests
├── requirements.txt # Python dependencies
├── README.md # Project overview and instructions
└── setup.py # Installation script
