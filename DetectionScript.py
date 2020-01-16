python3 -m pip install --upgrade pip

pip3 install tensorflow
pip3 install numpy
pip3 install scipy
pip3 install opencv-python
pip3 install pillow
pip3 install matplotlib
pip3 install h5py
pip3 install keras
pip3 install https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.1/imageai-2.0.1-py3-none-any.whl

from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image.jpg"), output_image_path=os.path.join(execution_path , "ObjectDetection.jpg"))


