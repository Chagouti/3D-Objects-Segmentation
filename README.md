# IntrA 3D Objects Classification
This project implements a machine learning pipeline to classify 3D medical objects, specifically vessels and aneurysms, from the IntrA dataset. The goal of this project is to classify 3D point cloud data representing medical structures using advanced neural networks like PointNet and its variants.

## Project Overview:
The IntrA dataset contains 3D representations of vessels and aneurysms. This project uses PointNet, a model specifically designed to process 3D point clouds, for binary classification of medical images. The project is built with PyTorch and optimized for GPU processing.

## Dataset:
The IntrA dataset provides 3D point clouds in OBJ format, representing medical structures like vessels and aneurysms. The point cloud data consists of sets of 3D coordinates that capture the geometry of these objects, along with additional information like surface normals.

## Usage:
To train the model using PointNet2:
```Shell
python train_classification.py --model pointnet2_cls_ssg --use_normals --log_dir pointnet2_ssg_wo_normals
```
To evaluate the trained model on the test dataset:
```Shell
python test_classification.py --model pointnet2_cls_ssg --use_normals --log_dir pointnet2_ssg_wo_normals
```
## Performance
| Model | Aneurysm Class Accuracy | Vessel Class Acuracy |
|--|--|--|
| PointNet (Pytorch with normal) |  65| 94.98|
| PointNet2_SSG (Pytorch with normal) |  **90.51**| **98.52**|
| PointNet2_MSG (Pytorch with normal) |  88.83| 97.95|
