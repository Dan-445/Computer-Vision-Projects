#!/bin/sh
#gunicorn --chdir /home/iosph/edekee-ml/yolov5 wisgi:app -w 2 --threads 2 -b 0.0.0.0:8080
gunicorn main:app -w 5 --threads 10 -b 0.0.0.0:8080