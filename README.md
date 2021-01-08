# Multi-Camera Human Tracking and Re-Identification (using video)
Simple model to track and re-identify individuals in different cameras/videos.

## Introduction
This project aims to track people in different videos accounting for different angles. 


The framework used to accomplish this task relies on MOT and ReID to track and re-identify ID's of humans, respectively.
The tracking can be completed using YOLO_v3 or YOLO_v4 and ReID relies on KaiyangZhou's Torchreid library.
  
## Installation  
Download [Anaconda](https://www.anaconda.com/products/individual) if it is not installed on your machine

Clone the repository
```python
git clone https://github.com/samihormi/Multi-Camera-Human-Tracking-and-Re-Identification
```
Create a project environment
```python
cd Multi-Camera-Human-Tracking-and-Re-Identification
conda create --name py37 python=3.7
conda activate py37
```
Install dependencies
```python
pip install -r requirements.txt
```
Install torch and torchvision based on the cuda version of your machine
```python
conda install pytorch torchvision cudatoolkit -c pytorch
```
## Convert models
Download the YOLO models for [YOLO_v3](https://drive.google.com/file/d/18fmQMegNsAzPte7tJeCxwf1iE8JUTQhQ/view?usp=sharing) and [YOLO_v4](https://drive.google.com/file/d/1w9furPagm3KytRW2uNooLcBoiYWDwbop/view?usp=sharing) and add them to /model_data/weights/

YOLO_v3
```python
python convert_y3.py model_data\weights\yolov3.weights model_data\models\yolov3.h5
```
YOLO_v4
```python
python convert_y4.py model_data\weights\yolov4.weights model_data\models\yolov4.h5
```

# Pre-trained models (.h5) (If you want to start right away)
Download the Keras models for [YOLO_v3](https://drive.google.com/file/d/1a7JI-A920lrdt6OKya-qCXx-5ZUWvkMg/view?usp=sharing) and [YOLO_v4](https://drive.google.com/file/d/1pwFo4aHKPi0ztpL5tEYaXIr8RltYYQeY/view?usp=sharing) and add them to \model_data\models\ 


Download either one of the following Torchreid models [1](https://drive.google.com/file/d/1EtkBARD398UW93HwiVO9x3mByr0AeWMg/view?usp=sharing),[2](https://drive.google.com/open?id=15Ayri_sHtrctJ1Zb8qERjvdi66y6QaI4) and add them to \model_data\models\ (you might have to change the path in reid.py)

## Demo 
 
You can try out your own videos by running demo.py.  
Under the directory \videos\output, the program will generate a video of the tracking, as well as a video of the tracking and ReID.  
You should specify the path of the videos and the version of YOLO you would like to use (v3 or v4) 

```python
python demo.py --videos videos\init\Double1.mp4 videos\init\Single1.mp4 --version v3
```

## Acknowledgement
This model is build on top of the incredible work done in the following projects:
  * https://github.com/nwojke/cosine_metric_learning
  * https://github.com/KaiyangZhou/deep-person-reid
  * https://github.com/Qidian213/deep_sort_yolov3
  * https://github.com/Ma-Dan/keras-yolo4
  * https://github.com/lyrgwlr/Human-tracking-multicam
