# Galaxy CounTR

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Lightning](https://img.shields.io/badge/lightning-792EE5?style=for-the-badge&logo=pytorchlightning&logoColor=white)
![Albumentations](https://custom-icon-badges.demolab.com/badge/albumentations-FFFFF0?style=for-the-badge&logo=albumentations&logoColor=white)

In this project we wanted to test the ability of the [CounTR (Transformer-based Generalised Visual Counting)](https://github.com/Verg-Avesta/CounTR) on a newly created different dataset to see if the performance would've been great also with noisy images and on objects with a very small patch size.

## The dataset (Galaxy Counting Dataset)

The dataset was made from scratch starting from a FITS Image and the related segmentation mask of 25 Mpx.
The Image was patched and for each object the central pixel was annotated in a numpy binary array, while for training starting, from the groundtruth array, a gaussian filter was applied after multiplying the pixels x60 in order to avoid sparsity.

The dataset has been made publicy available on Kaggle at the following link: [Galaxy Counting Dataset](https://www.kaggle.com/datasets/santurini/fits-images-for-object-counting-and-detection)

![004224](https://user-images.githubusercontent.com/91251307/211839089-3a7eaf74-1c41-4a32-ac3d-a7056354c7bc.png)
![004223](https://user-images.githubusercontent.com/91251307/211839103-5ad1f822-638e-4455-914c-ee4ca8698e5a.png)

## Training

### Original Paper Pipeline ([folder](code/paper/README.md))

In the original paper the model is trained on augmented images (mixing, random cropping, blending) and tested by cropping the image and summing the counts on the individual patches. 

Starting from the pretrained weights, performing image reconstruction, on DOTA and CARPK, the model was finetuned for 100 epochs on the Galaxy dataset.

### Galaxy-CounTR ([folder](code/Galaxy-CounTR/README.md))

In the personalized pipeline has been used a different augmentation pipeline (random flip, color jittering, Random Noise) and were tested multiple loss functions and Learning rate Schedulers.

The code has been fully re-implemented using PyLightning both for Training and Testing.

## Results and Report

All the results and observations have been reported in an exhaustive [report](Galaxy_CounTR.pdf) about the project, also the data preparation and ablation studies are reported in order to give a better understanding of the workflow.

For sake of curiosity this is the comparison between the model trained with the original pipeline vs our implementation:

![bad_prediction](https://user-images.githubusercontent.com/91251307/211838935-3f4b416b-b566-4853-8bcf-88007d003cca.png)
![good_prediction](https://user-images.githubusercontent.com/91251307/211838955-08d0e552-7aa6-49dd-ae4e-9b05b9958323.png)
