import cv2
import torch
import argparse
import numpy as np

from typing import List
import os
import json
import datetime
import time
import os.path
from os.path import join
from glob import glob
from datetime import datetime
import pandas as pd
from utilities.color_detection import color_detection_new
from utilities.datatypes import Db_Connector, Configuration, InferenceModel, SingletonClass, Bbox
# from utilities.json_files import making_json
from flask import Flask, request, jsonify
from utilities.common_functions import model_detection,detection_post_processing
from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker


from tools import generate_detections as gdet
from utilities.generals import print_args
from pathlib import Path
import sys

from utilities.json_files import JsonWriter
from utilities.plots import  Annotator
from utilities.video import VideoReader, VideoWriter


from deep_sort.detection import Detection
import matplotlib.pyplot as plt
# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]  # YOLOv5 root directory
# if str(ROOT) not in sys.path:
    # sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from enum import Enum
import boto3

class AppType(Enum):
    FLASK = 0
    VIDEO = 1

class TrackerConfig(metaclass=SingletonClass):
    def __init__(self,max_cosine_distance=0.4,
                 nn_budget=None,
                 nms_max_overlap=0.5,
                 deepsort_model_path=r'/content/drive/MyDrive/AI-engine/mars-small128.pb'
                 ,class_names = []
                 ):
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        self.nms_max_overlap = nms_max_overlap
        # initialize deep sort
        self.deepsort_model_path = deepsort_model_path
        # '/content/drive/MyDrive/AI-engine/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(self.deepsort_model_path, batch_size=1)
        # calculate cosine distance metric
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
        # initialize tracker
        self.tracker = Tracker(self.metric,class_names=class_names)
        self.cmap = plt.get_cmap('tab20b')
        self.colors = [self.cmap(i)[:3] for i in np.linspace(0, 1, 20)]
        self.class_names = InferenceModel.get_class_names()
        self.tracks_list =[]

    def update(self,detections,frame):
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
        # Call the tracker
        self.tracker.predict()
        self.tracker.update(detections)
        Bboxes = []
        # update tracks
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()
            Box_cls_obj= Bbox(bbox,)

            # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 30)),
                          (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])), color,
                          -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75,
                        (255, 255, 255), 2)


    def get_tracker_boxes(self):
        bboxes_from_tracks = []
        # update tracks
        self.tracks_list =[]
        for k_track in self.tracker.track_based_on_class_ids:
            tracks=self.tracker.track_based_on_class_ids[k_track]
            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                class_name = track.get_class()
                index = self.class_names.index(class_name)
                label = f"{class_name}-{track.track_id}"
                color_name = track.get_color()
                bbox_cls_id = Bbox(bbox,class_id=index,score=1,label=label,primary_color=color_name)
                bboxes_from_tracks.append(bbox_cls_id)
                self.tracks_list.append(track)


        return bboxes_from_tracks

    def update_based_class_id(self,detections,frame):

        # Call the tracker
        self.tracker.predict()
        self.tracker.update(detections)

def Bbox_to_Detection(bboxes:List[Bbox],trk,frame):
    scores =[]
    names =[]
    boxes=[]
    primary_colors=[]

    for bbox_id in bboxes:
        xmin,ymin,xmax,ymax=bbox_id.bbox
        width = xmax-xmin
        height = ymax-ymin
        boxes.append([xmin,ymin,width,height])
        names.append(bbox_id.label)
        scores.append(bbox_id.score)
        primary_colors.append(bbox_id.primary_color)

        # print(bbox_id.bbox)
    boxes = np.array(boxes)
    features = trk.encoder(frame, boxes)
    detections = [Detection(bbox, score, class_name, feature,primary_color) for bbox, score, class_name, feature,primary_color in
                  zip(boxes, scores, names, features,primary_colors)]


    # run non-maxima supression
    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(boxs, trk.nms_max_overlap, scores)
    detections = [detections[i] for i in indices]
    class_ids_based_detections ={}
    for detection in detections:

        if detection.class_name not in class_ids_based_detections:
            class_ids_based_detections[detection.class_name] =[]

        class_ids_based_detections[detection.class_name].append(detection)


    return detections,class_ids_based_detections


def start(body):
    cfg = Configuration()
    trk = TrackerConfig()
    inference_model = InferenceModel(cfg)
    # class_names = inference_model.class_names
    output_folder ="output_folder"

    db_obj = Db_Connector() if cfg.upload_to_db else None
    video_path = body['video_path']
#
    video_reader= VideoReader(video_path)
    aws_access_key_id = cfg.aws_access_key_id
    aws_secret_access_key = cfg.aws_secret_access_key
    bucket_name = cfg.bucket_name
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id,aws_secret_access_key=aws_secret_access_key)
    # video_writer = VideoWriter(video_reader,output_folder)
    json_writer = JsonWriter(video_reader,output_folder,custom_path=None,s3=s3,bucket_name=bucket_name)
    json_writer_track = JsonWriter(video_reader,output_folder,custom_path="tracker",s3=s3,bucket_name=bucket_name)
    video_writer = None if "video_write" not in body else ( VideoWriter(video_reader,output_folder) if body["video_write"] else None)
    win_name = "frame" if cfg.show_img else None
    annotator = Annotator(output_folder,win_name)

# i solved the issue of passing access key from postman .try testing the code
    while True:
        ret, frame = video_reader.stream.read()
        if ret:
            prediction=model_detection(frame)
            bboxes=detection_post_processing(prediction)

            #color_detection_new(frame,bboxes)

            detections,class_ids_based_detections=Bbox_to_Detection(bboxes,trk,frame)
            # initialize color map
            trk.update_based_class_id(class_ids_based_detections,frame)
            tracker_bboxes =trk.get_tracker_boxes()
            # trk.tracks_drawing(frame)
            # trk.update(detections,frame)
            annotator.tracks_drawing(frame,trk,cfg.save_cropped_img)

            annotator.show_frame(frame)
            json_writer.addToJson(bboxes,frame)
            json_writer_track.addToJson(tracker_bboxes,frame)

            if cfg.save_detected_img:
                annotator.write_jpeg()
            if cfg.show_img:
                annotator.show_img()
            if video_writer :
                video_writer.videowrite(frame)
        else:
            break

    if db_obj:
        db_obj.save_in_db(body, 'peggs', json_writer.output_json)

    json_writer.making_json()
    json_writer_track.making_json()

    res = dict()
    res['status'] = '200'
    res['result'] = json_writer_track.output_json
    print('====Mask_Fashionpedia_Result====', res)


    del video_reader
    if video_writer:
        del video_writer

    return res


flask_app = Flask(__name__)



@flask_app.route("/start", methods=['POST'])
def home():
    print(("json :", request.get_json()))
    if request.method == 'POST':
        body = request.get_json()
        res = start(body)
        return jsonify(res)

def run(
        app_type=0,
        host= "0.0.0.0",  # model path or triton URL
        port=3000,
        yolov5_repo_path=r"/home/jupyter/AI-engine/yolov5",
        yolov5_model_path = r"/home/jupyter/AI-engine/yolov5/models/fashionpedia_100_v1.pt",
        deepsort_model_path = r'mars-small128.pb',
        color_file = r"/home/jupyter/AI-engine/utilities/color_recognition/colors.csv",
        conf_threshold = 0.50,
        model_iou = 0.30,
        color_detection = False,
        upload_to_db = False,
        show_img = False,
        save_detected_img = False,
        save_cropped_img = False,
        db_host = "",
        db_user = "",
        db_password = "",
        db_name = "",
        pant_top_left=False,
        max_cosine_distance=0.10,
        nn_budget=None,
        nms_max_overlap=0.10,
        video_path =r"/content/drive/MyDrive/Test_videos/pexels-rodnae-productions-6192896.mp4",
        video_id="05149607-ccf5-401b-b81f-47e687c05ea1",
        video_write=True,
        aws_access_key_id=None,
        aws_secret_access_key=None,
        bucket_name =None
):
    global flask_app
    cfg = Configuration(
    yolov5_repo_path,yolov5_model_path,color_file,
    conf_threshold,model_iou ,color_detection ,
    upload_to_db ,show_img,save_detected_img ,save_cropped_img,db_host ,db_user
    ,db_password ,db_name ,pant_top_left,aws_access_key_id,aws_secret_access_key,bucket_name)

    inference_model = InferenceModel(cfg)
    class_names =inference_model.class_names
    trk = TrackerConfig(max_cosine_distance,nn_budget,nms_max_overlap,deepsort_model_path, class_names,
                        )

    app_type = AppType(app_type)
    if app_type == AppType.FLASK:
        flask_app.run(host=host, port=3000)
    else:
        flask_app =None
        body ={"video_path":video_path ,"video_id": video_id, "video_write": video_write }
        start(body)
    pass


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--app_type', type=int, default=0, help='App Type (0 for FLASK and 1 for Video')
    # Video Path if app type is 1
    parser.add_argument('--video_path', type=str, default=r"/home/jupyter/AI-engine/test_vids/pexels.mp4", help='video path for detection')
    # parser.add_argument('--video_id', type=str, default=r"05149607-ccf5-401b-b81f-47e687c05ea1", help='video id')
    parser.add_argument('--video_write', default=False, action='store_true', help='video write')
    # Flask Path if app type is 0
    parser.add_argument("--host", "-H", help="host name running server", type=str, required=False, default='localhost')
    parser.add_argument("--port", "-P", help="port of runnning server", type=int, required=False, default=3000)
    # Default Settings..
    parser.add_argument('--yolov5_repo_path', nargs='+', type=str, default=r"/home/jupyter/AI-engine/yolov5", help='model path or triton URL')
    parser.add_argument('--yolov5_model_path', type=str, default=r"/home/jupyter/AI-engine/yolov5/models/fashionpedia_100_v1.pt" , help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--color_file', type=str, default=r"/home/jupyter/AI-engine/utilities/color_recognition/colors.csv", help='file/dir/URL/glob/screen/0(webcam)')
    # parser.add_argument('--color_detection', default=False, action='store_true', help='hide labels')
    parser.add_argument('--upload_to_db', default=False, action='store_true', help='hide labels')
    parser.add_argument('--show_img', default=False, action='store_true', help='hide labels')
    parser.add_argument('--save_detected_img', default=False, action='store_true', help='hide labels')
    parser.add_argument('--save_cropped_img', default=True, action='store_true', help='Save cropped_images of detections')
    parser.add_argument('--conf_threshold', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--model_iou', type=float, default=0.30, help='NMS IoU threshold')
    parser.add_argument('--pant_top_left',  default=False, help='Pant_Type_Check')
    parser.add_argument('--db_name', type=str, default="", help='Database Name')
    parser.add_argument('--db_user', type=str, default="", help='Database User')
    parser.add_argument('--db_password', type=str, default="", help='Database Password')
        #Tracker Parameters
    parser.add_argument('--deepsort_model_path', type=str, default=r'mars-small128.pb' , help='Deep Sort Model Path')
    parser.add_argument('--max_cosine_distance', type=float, default=0.45, help='Max Cosine Distance Threshld')

    
    parser.add_argument('--aws_access_key_id', type=str, default='AKIAUPZNAO44EG6D7RTW', help='AWS ACCESS KEY ID')
    parser.add_argument('--aws_secret_access_key', type=str, default='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', help='AWS SECRET ACCESS KEY')
    parser.add_argument('--bucket_name', type=str, default='ai-engine-cropped-image-bucket', help='S3 Bucket name')


    opt = parser.parse_args()
    # opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt






if __name__ == "__main__":
    opt = parse_opt()
    run(**vars(opt))



# if __name__ == "__main__":
#     opt = parse_opt(
#     main(opt)
#
# if __name__ == "__main__":
#     host, port = importargs()
#
#     app.run(host="0.0.0.0", port=8081)
