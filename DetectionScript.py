# python3 -m pip install --upgrade pip
'''
pip3 install tensorflow
OR
python3 -m pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.12.0-py3-none-any.whl'
OR
pip3 install tensorflow==1.14.0
OR
pip3 install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.14.0-py3-none-any.whl (recommended)
pip3 install keras
OR
pip3 install keras==2.2.0python
OR
pip3 install keras==2.2.0 (recommended)
'''
# pip3 install numpy
# pip3 install scipy
# pip3 install opencv-python
# pip3 install pillow
# pip3 install matplotlib
# pip3 install h5py
# pip3 install https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.1/imageai-2.0.1-py3-none-any.whl

'''
DOWNLOAD RetinaNet from here:
https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5
'''

from imageai.Detection import ObjectDetection

# 20200227
# from keras import backend
# from tensorflow.keras import backend
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "object.jpg"), output_image_path=os.path.join(execution_path , "object_new.jpg"))

for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
