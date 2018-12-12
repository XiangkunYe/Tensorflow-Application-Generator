# TFModel(Tensorflow Model Training Module)

## dataset we used
1. Flowers Dataset  
Source: Tensorflow(http://download.tensorflow.org/example_images/flower_photos.tgz)  
Size: 4300+ images(330MB)  
Used for: Image Classification  
Description: This Dataset contains 6 different flower species. So there will have 6 folders each contains one kind of flowers' images. The images' quality is really hign, so this dataset is one of the datasets we used to test our TFModel module's training accuracy.  
Here's an example image of this dataset:  
![Alt text](images_for_README/flower_dataset_sample.jpg?raw=true "Title") 
2. Monkeys Dataset
Source: Kaggle(https://www.kaggle.com/slothkong/10-monkey-species/home)  
Size: 1400+ images(434MB)
Used for: Image Classification
Description: This Dataset contains 10 different Monkey species. It contains 10 subfolders labeled as n0~n9, each corresponding a species form Wikipedia's monkey cladogram. Images are 400x300 px or larger and JPEG format. This dataset is contains less images than the Flowers Dataset, but the images have higher quality.  
Here's an example image of this dataset:  
![Alt text](images_for_README/monkey_dataset_sample.jpg?raw=true "Title")  
3. Art Images Dataset
Source: Kaggle(https://www.kaggle.com/thedownhill/art-images-drawings-painting-sculpture-engraving)  
Size: 7500+ images(322MB)
Used for: Image Classification
Description: It's a dataset for classifying different styles of art. It includes 5 different kinds of art images. It's harder to classify than the 2 datasets above.  
Here's an example image of this dataset:  
![Alt text](images_for_README/art_dataset_sample.jpeg?raw=true "Title")  
4. Cars Dataset
Source: Tensorflow
Size: 1000 images(103MB)
Used for: Image Segmentation
Description: This dataset is used for image segmentation. It contains two folders: images and labels. In 'images', there are about 1000 car images. In 'labels', there are 1000 car's label images. Each represents as a label of one image from the 'images' folder.  
Here's an example image of this dataset:  
![Alt text](images_for_README/car_dataset_sample_image.jpg?raw=true "Title")
Here's an exmaple image of the label image:  
![Alt text](images_for_README/car_dataset_sample_label.gif?raw=true "Title")  

## model we used
1. Inception V3 Model  
Source: Keras(Open-Source)  
License: MIT  
Structure:  
![Alt text](images_for_README/v3.png?raw=true "Title")  
Description:  
Inception v3 is a widely-used image recognition model that has been shown to attain greater than 78.1% accuracy on the ImageNet dataset. The model is the culmination of many ideas developed by multiple researchers over the years. It is based on the original paper: "[Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)" by Szegedy, et. al.  
In TFModel, we use inception V3 model as our image classification model and retrain the last layer per user's requirements. This method is called 'Transfer Learning', which can greatly improve the accuracy of the model. Transfer learning is a technique that shortcuts much of this by taking a piece of a model that has already been trained on a related task and reusing it in a new model. The magic of transfer learning is that lower layers that have been trained to distinguish between some objects can be reused for many recognition tasks without any alteration.  
Here we compare the validation accuracy and training accuracy between the Inception V3 Model and a CNN Model designed by ourselves on the Monkeys Dataset(50 epoches):  
![Alt text](images_for_README/compare.png?raw=true "Title")  
We found out that using inception V3 Model is much faster to converge and can achieve higher accuracy.  
2. Unet Model  
Source: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)  
Structure:  
![Alt text](images_for_README/unet.png?raw=true "Title")  
Description:  
The u-net is convolutional network architecture for fast and precise segmentation of images. Up to now it has outperformed the prior best method in the area of Image Segmentation.  
The TFModel will train Unet Model per user's requirements.  