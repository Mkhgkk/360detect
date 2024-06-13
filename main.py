
'''
Object Detection on Panorama pictures
Usage:
    $ python3 detection.py <pano_picture> <output_picture>

    pano_picture(str)  : the pano pic file
    output_picture(str): the result picture
'''
import sys
import cv2
import numpy as np
from ultralytics import YOLO
from stereo import pano2stereo, realign_bbox
from math import pi, atan
from OpenGL.GL import *
from OpenGL.GLU import *
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtOpenGL import QGLWidget
from PIL import Image
from EquiView360 import MainWindow
from PIL import Image


CF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
INPUT_RESOLUTION = (640, 640)  # Adjust for YOLOv8

class Yolo():
    '''
    YOLOv8 object detection class
    '''
    def __init__(self, model_path='yolov8n.pt'): # You can change model here
        self.model = YOLO(model_path)
        self.cf_th = CF_THRESHOLD
        self.nms_th = NMS_THRESHOLD
        self.resolution = INPUT_RESOLUTION
        print('Model Initialization Done!')

    def detect(self, frame):
        '''
        Perform object detection using YOLOv8

        Args:
            frame (np.array): Input image for object detection

        Returns:
            results (ultralytics.yolo.engine.results.Results): YOLOv8 results object
        '''
        results = self.model(frame, conf=self.cf_th, iou=self.nms_th,classes=[0])
        return results
    

    def draw_bbox(self, frame, class_id, conf, left, top, right, bottom):
        '''
        Draw a bounding box on the image

        Args:
            frame (np.array): The image to draw on
            class_id (int): Class ID of the detected object
            conf (float): Confidence score of the detection
            left (int): Left x-coordinate of the bounding box
            top (int): Top y-coordinate of the bounding box
            right (int): Right x-coordinate of the bounding box
            bottom (int): Bottom y-coordinate of the bounding box

        Returns:
            frame (np.array): Image with bounding box drawn
        '''
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

        label = f'{self.model.model.names[class_id]} {conf:.2f}'

        # Display the label at the top of the bounding box
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, label_size[1])
        cv2.rectangle(frame,
                      (left, top - round(1.5*label_size[1])),
                      (left + round(label_size[0]), top + base_line),
                      (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return frame


    def process_output(self, input_img, frames):
        '''
        Process YOLOv8 results and draw bounding boxes on the panorama image

        Args:
            input_img (np.array): The original panorama image
            frames (list): List of four perspective views from pano2stereo

        Returns:
            base_frame (np.array): Panorama image with bounding boxes drawn
        '''
        height = frames[0].shape[0]
        width = frames[0].shape[1]
        base_frame = input_img.copy()
                
        def combine_overlapping_bounding_boxes(bounding_boxes):
            combined_boxes = bounding_boxes.copy()  # Create a copy of the original list

            # Iterate through each pair of bounding boxes
            for i in range(len(combined_boxes)):
                for j in range(i+1, len(combined_boxes)):
                    box1 = combined_boxes[i]
                    box2 = combined_boxes[j]

                    # Check for overlap
                    if box1[0] <= box2[2] and box1[2] >= box2[0] and box1[1] <= box2[3] and box1[3] >= box2[1]:
                        # Calculate coordinates of the combined bounding box
                        combined_x1 = min(box1[0], box2[0])
                        combined_y1 = min(box1[1], box2[1])
                        combined_x2 = max(box1[2], box2[2])
                        combined_y2 = max(box1[3], box2[3])

                        # Replace the original bounding boxes with the combined bounding box
                        combined_boxes[i] = [combined_x1, combined_y1, combined_x2, combined_y2]
                        combined_boxes.pop(j)

                        # Restart the iteration since the list has changed
                        break

            return combined_boxes







        print('Yolo Detecting...')
        bouding_box = []
        for face, frame in enumerate(frames):
            results = self.detect(frame)
            boxes = results[0].boxes

            for box in boxes:
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                bbox = box.xyxy[0].tolist()
                
                center_x = (bbox[0] + bbox[2]) / (2 * width) 
                center_y = (bbox[1] + bbox[3]) / (2 * height)
                box_width = (bbox[2] - bbox[0]) / width
                box_height = (bbox[3] - bbox[1]) / height

                center_phi, center_theta, pano_width, pano_height = realign_bbox(center_x, center_y, box_width, box_height, face)

                # Convert from normalized coordinates to pixel coordinates
                left = int((center_phi - pano_width / 2) * input_img.shape[1])
                top = int((center_theta - pano_height / 2) * input_img.shape[0])
                right = int((center_phi + pano_width / 2) * input_img.shape[1])
                bottom = int((center_theta + pano_height / 2) * input_img.shape[0])
                
                bouding_box.append([left, top, right, bottom])
                # print(bouding_box)
        overlapping_boxes = combine_overlapping_bounding_boxes(bouding_box)
        print(overlapping_boxes)
        for i in overlapping_boxes:
            left1, top1, right1, bottom1 = i
            self.draw_bbox(base_frame, cls, conf,  left1, top1, right1, bottom1)

        return base_frame

def main():
    '''
    Main function for testing
    '''
    if len(sys.argv) != 3:
        print("Usage: python detection.py <pano_picture> <output_picture>")
        return

    input_image = sys.argv[1]
    output_image = sys.argv[2]

    my_net = Yolo()
    input_pano = cv2.imread(input_image)
    projections = pano2stereo(input_pano)

    output_frame = my_net.process_output(input_pano, projections)
    cv2.imwrite(output_image, output_frame)
    # app = QtWidgets.QApplication([])
    # window = MainWindow()
    # window.gl_widget.image = Image.open(output_image) #don't change it
    # window.setGeometry(0, 0, 1600, 900)
    # window.show()
    # sys.exit(app.exec_())

if __name__ == '__main__':
    main()