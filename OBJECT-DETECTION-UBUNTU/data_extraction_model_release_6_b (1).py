#!/usr/bin/env python
# -*- coding: utf-8 -*-

## Import Libraries #################################################################################################################

import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, Markup
from werkzeug.utils import secure_filename
import subprocess
import cv2
import numpy as np
import argparse
import time
import re
import base64
import io
from PIL import Image
import re
from imageio import imread
import csv
import json
from imutils.object_detection import non_max_suppression
import math
from random import randint
from statistics import mode 
from flask import make_response
from functools import wraps, update_wrapper
from datetime import datetime
import itertools
import pytesseract
from statistics import StatisticsError

#### Initialise Variables ######################################################################################################################## 

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'  ##location of tesseract file,  not required if path varible s

#### Flask Setup ##################################################################################################################### 
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg', 'bmp'])
app.config['OUTPUT_FOLDER'] = 'output/'
#app.config['FILES_FOLDER'] = 'TrainingFiles/'


#weights_file =  "WP.weights"
#cfg_file = "experiment.cfg"
weights_file =  "yolov3-voc-WP_31_10000.weights"
cfg_file = "yolov3-voc-WP_31_resized.cfg"
names_file = "Combined_v6.names" ####

iou_threshold = 0.2
conf_threshold = 0.25 ## threshold for showing detections

conf_al_threshold = 0.70

 
symbol=1
t=1
pipeline=1
pipes=[]
#totalItems = 0
highestXS = 0
lowestXS = 0
highestYX = 0
lowestYX = 0
#extra = 150
#textJoin =[]
#sensorD = []
xC = 0
yC = 0
wC = 0
hC = 0
pX = 0
pY = 0
aa=1
currentPID = "blank"
imageShown =  "image"
qt=0

isCropped = 0
filenamePIDTimeS = ''
nameForCSVFile = ''
nameForCSVFileSymbols = ''
nameForCSVFileSymbols_c = ''

itemNo = 0 #csvNumber
detectNo = 0  ##displayNumber
addedSName =[]
removedSName = []

lines = []
pipelines = []
g,k,m = 1,1,1
a,b,c = 1,1,1

lines2 = []
d,e,f = 1,1,1

results = []
boxesPID = [] 
boxesPID_updated = []

boxesFull = []
confidencesFull = []
classIDsFull = []

COLORS=[]
LABELS=[]
arrows=[]
C1 = 0
C2 = 0
C3 = 0
scanAreaY = []
scanAreaX = []
#fTime = ""
#adjust = 8
#lineSizesH = []
#lineSizesV = []

origH = 7000
origW = 7000

symbolCount = 0
itemList = []
pipesDetected = []

drawingNumber = ""
numberBox = []

drawingTitle = ""
titleBox = []

xGrid = 0
yGrid = 0
filePresent = 0
annotationsStoreList = []
nameForChangeCSV = ''
symbolList = []
symbolRemove = -1
symbolClassList = []
symbolClassSelected = 0
symbolClassIndex = 0
removedSymbols = []
changedSymbols = []
addedSymbols = []
changedSymbolsName = []
approvedSymbols = []
panX = 0 
panY = 0
sf = 1
symbolNumber = 0

dashes=[]
lineSizesH = []
lineSizesV = []
nameForCSVFileText = ''
nameForCSVFilePipelines = ''
nameForCSVFileTextOnly = ''
totalSymbols = 0
totalSymbolsText = 0
textJoin =[]
totalItems = 0

high_pred = []
low_pred = []
manual_checked_symbols = []

## Functions ###################################################################################################################################
    
## allowed_file function ###################################################################################################################################
## function to confirm if filetype is within the list of allowed files ###################################################################################################################################    
def allowed_file(filename):
    return '.' in filename and \
           (filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS'] or
       filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS'])
            

## drawNumber function ###################################################################################################################################
## function to draw a number on the specified drawing  ################################################################################################################################### 
def drawNumber(image, text, x, y):
    
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)    


## pointBox function ###################################################################################################################################
## function to determine if specified point is within symbols bounding box ###################################################################################################################################
def pointBox(x,y):
    
    border = 30
    inBox = 0
    
    for c in range (0, len(boxesPID)):
        
        yLb = boxesPID[c][1]
        yHb = boxesPID[c][1] + boxesPID[c][3]
        
        xLb = boxesPID[c][0]
        xHb = boxesPID[c][0] + boxesPID[c][2]
        
        if ((xLb-border) <= x <= (xHb + border)) and ((yLb-border) <= y <= (yHb + border)):
            
            inBox = 1
            
    return inBox

## findDimension function ###################################################################################################################################
## function to findDimension ###################################################################################################################################
def findDimension(x1, y1, x2, y2, t):

    text_i_ = ['GS', 'GR', 'RS', 'RD', 'RL', 'HWS', 'HWR', 'RWS', 'RWR', 'CHWS', 'CHWR', 'CWS', 'CWR', 'FOS', 'FOR']

    text_i = sorted(text_i_, key=len, reverse=True)

    dimensionText = []

    for ((startX, startY, endX, endY), text) in results: 
        text = "".join([c for c in text]).strip()  

        sub = text.split()

        code = ''
        sub_d = ''

        for a, word in enumerate(sub):
            for item in text_i: 
                if item in word:
                    if (a != 0):
                        r = sub[a-1]
                        code = item
                    ## extract radius
                        for i, c in enumerate(r):
                            if c.isdigit() and (c != '0'):

                                if (len(r) > i+3): ## check
                                    if len(r)>(i+3): ### check for 3 digit measures
                                        if r[i+1].isdigit() and r[i+2].isdigit():
                                            if (r[i+2] == '0') or (r[i+2] == '5'):
                                                sub_d = c + r[i+1] + r[i+2]
                                                text_dim = sub_d + ' ' + code
                                                dimensionText.append( [(startX, startY, endX, endY), text_dim] )
                                            else:
                                                sub_d = c + r[i+1]
                                                text_dim = sub_d + ' ' + code
                                                dimensionText.append( [(startX, startY, endX, endY), text_dim] )

                                elif (len(r) == i+3): ## assume 2 digits
                                #elif len(r)>(i+1): ### check for 2 digit measures
                                    if r[i+1].isdigit():
                                        if (r[i+2] == '0') or (r[i+2] == '5'):
                                            sub_d = c + r[i+1]
                                            text_dim = sub_d + ' ' + code
                                            dimensionText.append( [(startX, startY, endX, endY), text_dim] )

    
    distY = 150000
    distX = 150000
    dim = ''

    if ( (y2-y1) > (x2-x1) ) : 

        for d in dimensionText:

            if ( (x1-t) <= d[0][0] <= (x2+t) ):

                if (abs(d[0][1]-y1)) < distY:
                    dim = d[1]

    else:

        for d in dimensionText:

            if ( (y1-t) <= d[0][1] <= (y2+t) ):

                if (abs(d[0][0]-x1)) < distX:
                    dim = d[1]

    a = ''

    for i, c in enumerate(dim):
        if c.isdigit():
            a = dim[i]

            try:
                #dim[i+1]
                if dim[i+1].isdigit() or dim[i+1] == 'O' or dim[i+1] == 'o':
                    a = a + dim[i+1]
            except IndexError:
                break

            break


    dim_list = []
    dim_list = dim.split()

    if (len(dim_list) == 2):
        return dim_list[0], dim_list[1]
    else:
        return dim_list, dim_list

#### nearestLine
def findDimensionS(x1, y1, x2, y2):


    b, dm = findDimension(x1, y1, x2, y2, 1000)

    


    return b

## drawPipelines function ###################################################################################################################################
## function to draw pipelines on ImageInput ################################################################################################################################### 
def drawPipelines(imageInput):
        
        print('drawPipelines')
        
        global nameForCSVFile       
        global itemNo       
        global totalSymbolsText
        global currentPID
        global adjust
        global pipelines
        global pipes
        global dashes
        global symbolCount 
        global itemList
        
        print("item count before symbols drawn on PID:", symbolCount)
        print("length of pipelines in drawPipelines is:", len(pipes))
        print("length of dashes in drawPipelines is:", len(dashes))        
        pipelineCounter = totalSymbolsText    
        print("pipelineCounter number prior to drawing pipelines:", pipelineCounter)
     
        filenameImageA2 = currentPID[:-4] + "_A2.png" ##pre-processed image
        imageInput2 = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameImageA2))
        imageInput3 = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameImageA2))        
        
        for n in range(0, len(dashes)): 
  
           #cv2.rectangle(imageInput, (dashes[n][0], dashes[n][1]), (dashes[n][2], dashes[n][3]), (0,255,0), -1)
           cv2.rectangle(imageInput2, (dashes[n][0], dashes[n][1]), (dashes[n][2], dashes[n][3]), (0,255,0), -1)
           cv2.rectangle(imageInput3, (dashes[n][0], dashes[n][1]), (dashes[n][2], dashes[n][3]), (255,255,255), -1)
                
        for i in range(0, len(pipes)): 
            
           cv2.rectangle(imageInput, (pipes[i][0], pipes[i][1]), (pipes[i][2], pipes[i][3]), (255,0,0), -1)
           cv2.rectangle(imageInput2, (pipes[i][0], pipes[i][1]), (pipes[i][2], pipes[i][3]), (255,0,0), -1)       
           cv2.rectangle(imageInput3, (pipes[i][0], pipes[i][1]), (pipes[i][2], pipes[i][3]), (255,255,255), -1)
                
           pipelineCounter = pipelineCounter + 1       
           #itemList.append([symbolCount, pipes[i][0], pipes[i][1], pipes[i][2], pipes[i][3], "pipeline", "pipeline"])
           text = str(pipelineCounter)
           drawNumber(imageInput, text, pipes[i][0], (pipes[i][1] - 20))    

        print("pipelineCounter after drawPipelines is:", pipelineCounter)
        print("item count after drawPipelines is:", symbolCount)
        #print("itemList after pipelines drawn on PID:", len(itemList))
        
        file1 = currentPID[:-4] + "_pipes.png"
        cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], file1), imageInput2) 
        
        file2 = currentPID[:-4] + "_A3.png"
        cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], file2), imageInput3)  

## dashesCheckText function ###################################################################################################################################
## function to remove dashes within text boxes  ################################################################################################################################### 
def dashesCheckText(dashes):##removes 'dashes' within text box
    
    global boxesPadding
    global boxesPID
    dashes3 = []

    h = 0
    v = 0
    print("start length dashes before check text ", len(dashes), len(dashes3))
    
    filenameToRead = currentPID[:-4] + "_detection.png"   
    im = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameToRead)) 
    
    for i in range(0, len(dashes)):
        
        inBox = 0
        
        y1T = dashes[i][1] + yC
        y2T = dashes[i][3] + yC
        x1T = dashes[i][0] + xC
        x2T = dashes[i][2] + xC
        
        cv2.rectangle(im, (x1T, y1T), (x2T, y2T), (0, 255, 0), 5)

        for (startX, startY, endX, endY) in boxesPadding:

            startX = int(startX)  #for whole image
            startY = int(startY)
            endX = int(endX)
            endY = int(endY)
            
            cv2.rectangle(im, (startX, startY), (endX, endY), (255, 0, 0), 4)
            
            if(startX<=x1T<=endX) and (startY<=y1T<=endY):
                if(startX<=x2T<=endX) and (startY<=y2T<=endY):##line is within the text box
                    inBox = 1
                    
                    cv2.rectangle(im, (x1T, y1T), (x2T, y2T), (255, 255, 0), 2)

        if (inBox == 0):
            dashes3.append([dashes[i][0], dashes[i][1], dashes[i][2], dashes[i][3]])
            
            
            
    name3 = "dashes.png"
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], name3), im) 
    
    print("end length dashes", len(dashes))                        
    print("length dashes3", len(dashes3))   
  
    return dashes3    

## dashesCheckText function ###################################################################################################################################
## function to remove dashes within text boxes  ################################################################################################################################### 
def dashesCheckText(dashes):##removes 'dashes' within text box
    
    global boxesPadding
    global boxesPID
    dashes3 = []

    h = 0
    v = 0
    print("start length dashes before check text ", len(dashes), len(dashes3))
    
    filenameToRead = currentPID[:-4] + "_detection.png"   
    im = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameToRead)) 
    
    for i in range(0, len(dashes)):
        
        inBox = 0
        
        y1T = dashes[i][1] + yC
        y2T = dashes[i][3] + yC
        x1T = dashes[i][0] + xC
        x2T = dashes[i][2] + xC
        
        cv2.rectangle(im, (x1T, y1T), (x2T, y2T), (0, 255, 0), 5)

        for (startX, startY, endX, endY) in boxesPadding:

            startX = int(startX)  #for whole image
            startY = int(startY)
            endX = int(endX)
            endY = int(endY)
            
            cv2.rectangle(im, (startX, startY), (endX, endY), (255, 0, 0), 4)
            
            if(startX<=x1T<=endX) and (startY<=y1T<=endY):
                if(startX<=x2T<=endX) and (startY<=y2T<=endY):##line is within the text box
                    inBox = 1
                    
                    cv2.rectangle(im, (x1T, y1T), (x2T, y2T), (255, 255, 0), 2)

        if (inBox == 0):
            dashes3.append([dashes[i][0], dashes[i][1], dashes[i][2], dashes[i][3]])
            
            
            
    name3 = "dashes.png"
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], name3), im) 
    
    print("end length dashes", len(dashes))                        
    print("length dashes3", len(dashes3))   
  
    return dashes3    

## pipesCheckText2 function ###################################################################################################################################
## function to discards any lines within the same text box  ################################################################################################################################### 
def pipesCheckText2(pipes):##discards any lines only within the same text box
    
    global boxesPadding
    global boxesPID
    pipes3 = []

    h = 0
    v = 0
    print("start length pipes before check text ", len(pipes), len(pipes3))
    
    filenameToRead = currentPID
    im = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameToRead)) 
    
    for i in range(0, len(pipes)):
        
        inBox = 0
        
        y1T = pipes[i][1] + yC
        y2T = pipes[i][3] + yC
        x1T = pipes[i][0] + xC
        x2T = pipes[i][2] + xC
        
        cv2.rectangle(im, (x1T, y1T), (x2T, y2T), (0, 255, 0), 5)

        for (startX, startY, endX, endY) in boxesPadding:

            startX = int(startX)  #for whole image
            startY = int(startY)
            endX = int(endX)
            endY = int(endY)
            
            cv2.rectangle(im, (startX, startY), (endX, endY), (255, 0, 0), 4)
            
            if(startX<=x1T<=endX) and (startY<=y1T<=endY):
                if(startX<=x2T<=endX) and (startY<=y2T<=endY):##line is within the text box
                    inBox = 1
                    
                    cv2.rectangle(im, (x1T, y1T), (x2T, y2T), (255, 255, 0), 2)

        if (inBox == 0):
            pipes3.append([pipes[i][0], pipes[i][1], pipes[i][2], pipes[i][3]])
            
    name3 = currentPID[:-4] + "_after_pipesCheckText.png"
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], name3), im) 
    
    print("2nd end length pipes", len(pipes))                        
    print("2nd length pipes3", len(pipes3))   
  
    return pipes3   


## pipesCheck function ###################################################################################################################################
## function to check if pipes are near symbols bounding box  ################################################################################################################################### 
def pipesCheck(): 
    
    global pipes
    global boxesPID
    global arrows
    global results
    pipes2 = []
    pipes4 = []
    boxesPipes = []
    pipeInterest = []

    margin = 50
    text_i = ['GS', 'GR', 'RS', 'RD', 'RL', 'HWS', 'HWR', 'RWS', 'RWR', 'CHWS', 'CHWR', 'CWS', 'CWR', 'FOS', 'FOR']

    h = 0
    v = 0
    print("start length pipes, arrows, h, v", len(pipes), len(arrows), h, v)
    
    filenameToRead = currentPID[:-4] + "_detection.png"   
    im = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameToRead)) 
    im2 = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameToRead))     
    im3 = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameToRead))  
    im4 = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameToRead))      
    im6 = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameToRead)) 
    im7 = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameToRead))   
    im8 = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameToRead))  
    im100 = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameToRead)) 
    im200 = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameToRead)) 
    im300 = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameToRead)) 

    linesToMeasure = []
    spacesMargin = 200
    thicknessMargin = 10
    margin2 = 200


    for i in range(0, len(pipes)):
        
        toAdd=1
        
        y1T = pipes[i][3] + yC
        y2T = pipes[i][1] + yC
        x1T = pipes[i][0] + xC
        x2T = pipes[i][2] + xC

        wT = abs(x2T-x1T)  
        hT = abs(y2T-y1T)  
        
        cv2.rectangle(im2, (x1T, y1T), (x2T, y2T), (0, 0, 255), -1)

        if (wT>hT):#if horizontalif()
            h = h + 1
            border = 0.05 * wT
            border = int(border)
            #print("border i", border)            
            cv2.rectangle(im, (pipes[i][0]+xC, pipes[i][1]+yC), (pipes[i][2]+xC, pipes[i][3]+yC), (0, 0, 255), 20)

            for ((startX, startY, endX, endY), text) in results:
                #text = "".join([c if ord(c) < 128 else "" for c in text]).strip()   
                text = "".join([c for c in text]).strip()  
                for item in text_i: # for each code of interest that is in text_i, 
                    if len([x for x in text if x.isdigit()]) >= 1: ## check the recognised text has at least one digit
                        if item in text: #if the detected text is in the list of codes of interest 
                            if ( ( pipes[i][1]+yC - margin) <= startY <= (pipes[i][3]+yC + margin) ) and ( ( pipes[i][1]+yC - margin) <= endY <= (pipes[i][3]+yC + margin) ): #if line is within a certain Y margin of the code of interest,
                                cv2.rectangle(im100, (pipes[i][0]+xC, pipes[i][1]+yC), (pipes[i][2]+xC, pipes[i][3]+yC), (0, 0, 255), 20) # then highlight the line on im100

                                ## create a list of the lines
                                linesToMeasure.append([pipes[i][0]+xC, pipes[i][1]+yC, pipes[i][2]+xC, pipes[i][3]+yC])


            tempY1 = pipes[i][1] + yC  
            tempX1 = pipes[i][0] + xC
            tempX2 = pipes[i][2] + xC
            tempY2 = pipes[i][3] + yC
            
            for b in range (0, len(boxesPID)):
                
                yLb = boxesPID[b][1]
                yHb = boxesPID[b][1] + boxesPID[b][3]
                
                xLb = boxesPID[b][0]
                xHb = boxesPID[b][0] + boxesPID[b][2]
                s=0
                if (tempY1>yLb) and (tempY2<yHb): 

                    cv2.rectangle(im3, (tempX1, tempY1), (tempX2, tempY2), (255, 0, 255), 15)
                    lowX = min(tempX1, tempX2)
                    highX = max(tempX1, tempX2) 
                    
                    if ((xLb-border) <= lowX <= (xHb + border)) and (toAdd == 1):  ##check for point 1  
                        #print("match point 1")
                        cv2.rectangle(im7, (tempX1, tempY1), (tempX2, tempY2), (255, 0, 255), 15)
                        cv2.rectangle(im8, (tempX1, tempY1), (tempX2, tempY2), (255, 0, 0), -1)
                        pipes2.append([tempX1, tempY1, tempX2, tempY2])
                        pipes4.append([tempX1, tempY1, tempX2, tempY2])   
                        toAdd = 0
                        
                        if not (pointBox(highX, tempY2)==1): ##check if point 2 is within a symbol box  (pointBox returns 1 if point is within a symbol box)
                            
                            boxesPx1 = tempX2 - int(hT*1)
                            boxesPy1 = tempY2 - int(hT*1)
                            
                            boxesPipes.append([boxesPx1, boxesPy1, (boxesPx1 +(hT*3)), (boxesPy1+(hT*3))])
                            cv2.rectangle(im7, (boxesPx1, boxesPy1), (boxesPx1 +(hT*3), boxesPy1+(hT*3)), (0, 255, 0), -1)                              
                        #
 

                    elif ((xLb-border) <= highX <= (xHb + border)) and (toAdd == 1):  ##check for point 2
                        #print("match point 2")
                        cv2.rectangle(im7, (tempX1, tempY1), (tempX2, tempY2), (255, 0, 255), 15)
                        cv2.rectangle(im8, (tempX1, tempY1), (tempX2, tempY2), (255, 0, 0), -1)
                        pipes2.append([tempX1, tempY1, tempX2, tempY2])
                        pipes4.append([tempX1, tempY1, tempX2, tempY2])
                        toAdd = 0  
                        
         


                        if not (pointBox(lowX, tempY1)==1): ##check if point 1 is within a symbol box
                            
                            boxesPx1 = tempX1 - int(hT*1)
                            boxesPy1 = tempY1 - int(hT*1)
                            
                            boxesPipes.append([boxesPx1, boxesPy1, (boxesPx1 +(hT*3)), (boxesPy1+(hT*3))])
                            cv2.rectangle(im7, (boxesPx1, boxesPy1), (boxesPx1 +(hT*3), boxesPy1+(hT*3)), (255, 0, 255), 2)    

                
                        
            
                        
              #  pipes.remove(pipes[i][0], pipes[i][1], pipes[i][2], pipes[i][3])

        elif(wT<hT):
            v = v + 1    

            border = 0.05 * hT
            border = int(border)
            #print("border i", border)            
            cv2.rectangle(im, (pipes[i][0]+xC, pipes[i][1]+yC), (pipes[i][2]+xC, pipes[i][3]+yC), (255, 0, 0), 20)

            for ((startX, startY, endX, endY), text) in results:
                #text = "".join([c if ord(c) < 128 else "" for c in text]).strip()   
                text = "".join([c for c in text]).strip()  
                for item in text_i:
                    #if item in text:

                    if  (' ' + item + ' ') in (' ' + text + ' ') :
                        if ( ( pipes[i][0]+xC - margin) <= startX <= (pipes[i][2]+xC + margin) ) and ( ( pipes[i][0]+xC - margin) <= endX <= (pipes[i][2]+xC + margin) ):
                            cv2.rectangle(im100, (pipes[i][0]+xC, pipes[i][1]+yC), (pipes[i][2]+xC, pipes[i][3]+yC), (255, 0, 0), 20)


                            ## create a list of the lines
                           # linesToMeasure.append([pipes[i][0]+xC, pipes[i][1]+yC), (pipes[i][2]+xC, pipes[i][3]+yC])


            
            tempY1 = pipes[i][1] + yC
            tempX1 = pipes[i][0] + xC
            tempX2 = pipes[i][2] + xC
            tempY2 = pipes[i][3] + yC
            
            for b in range (0, len(boxesPID)):
                
                yLb = boxesPID[b][1]
                yHb = boxesPID[b][1] + boxesPID[b][3]
                
                xLb = boxesPID[b][0]
                xHb = boxesPID[b][0] + boxesPID[b][2]
                
                if (tempX1>xLb) and (tempX2<xHb):##in range
                    
                    cv2.rectangle(im3, (tempX1, tempY1), (tempX2, tempY2), (255, 0, 255), 15)
                    lowY = min(tempY1, tempY2)
                    highY = max(tempY1, tempY2) 
                    
                    if ((yLb - border) <= lowY <= (yHb + border)) and (toAdd == 1):  ##check for point 1
                        #print("match point 3")
                        cv2.rectangle(im7, (tempX1, tempY1), (tempX2, tempY2), (255, 0, 255), 15)
                        cv2.rectangle(im8, (tempX1, tempY1), (tempX2, tempY2), (255, 0, 0), -1)
                        pipes2.append([tempX1, tempY1, tempX2, tempY2])
                        pipes4.append([tempX1, tempY1, tempX2, tempY2])
                        toAdd = 0
                        
                        
                        if not (pointBox(highX, tempY2)==1): ##check if point 2 is within a symbol box
                            
                            boxesPx1 = tempX2 - int(wT*2)
                            boxesPy1 = tempY2 - int(wT*2)
                            
                            boxesPipes.append([boxesPx1, boxesPy1, (boxesPx1 +(wT*3)), (boxesPy1+(wT*3))])
                            cv2.rectangle(im7, (boxesPx1, boxesPy1), (boxesPx1 +(wT*3), boxesPy1+(wT*3)), (255, 255, 0), 2)  


                            
                    elif ((yLb - border) <= highY <= (yHb + border)) and (toAdd == 1):  ##check for point 2
                        #print("match point 4")
                        cv2.rectangle(im7, (tempX1, tempY1), (tempX2, tempY2), (255, 0, 255), 15)
                        cv2.rectangle(im8, (tempX1, tempY1), (tempX2, tempY2), (255, 0, 0), -1)
                        pipes2.append([tempX1, tempY1, tempX2, tempY2])
                        pipes4.append([tempX1, tempY1, tempX2, tempY2])
                        toAdd = 0            
                        
                        if not (pointBox(lowX, tempY1)==1): ##check if point 1 is within a symbol box
                            
                            boxesPx1 = tempX1 - int(wT*1)
                            boxesPy1 = tempY1 - int(wT*1)
                            boxesPipes.append([boxesPx1, boxesPy1, (boxesPx1 +(wT*3)), (boxesPy1+(wT*3))])
                            cv2.rectangle(im7, (boxesPx1, boxesPy1), (boxesPx1 +(wT*3), boxesPy1+(wT*3)), (255, 255, 0), 2)   
                            
                            
                            
        #if (wT>hT):#if horizontalif()                        
        # #elif(wT<hT):                        
        # if (toAdd==1): #if not added
        
        #     for b in range (0, len(arrows)):
            
        #         xBorder = int(2.5* (arrows[b][2] - arrows[b][0]))   
        #         yBorder = int(2.5* (arrows[b][3] - arrows[b][1]))    
                
        #         yLb = arrows[b][1]
        #         yHb = arrows[b][3]
                
        #         xLb = arrows[b][0]
        #         xHb = arrows[b][2]
                
        #         cv2.rectangle(im2, (xLb-xBorder, yLb-yBorder), (xLb+xBorder, yLb+yBorder), (255, 0, 0), 3)
        #         cv2.rectangle(im2, (tempX1, tempY1), (tempX2, tempY2), (255, 0, 0), 10)
        #         if ((xLb-xBorder) <= tempX1 <= (xLb+xBorder)) and ((yLb-yBorder) <= tempY1 <= (yLb+yBorder)) and (toAdd == 1):
        #        # if ((xLb-border) <= tempX1 <= (xHb + border)) and ((yLb-border) <= tempY1 <= (yHb + border)) and (toAdd == 1):                    #print("line next to arrow 1")
        #             cv2.rectangle(im7, (tempX1, tempY1), (tempX2, tempY2), (255, 255, 0), 5)
        #             cv2.rectangle(im8, (tempX1, tempY1), (tempX2, tempY2), (255, 0, 0), -1)
        #             pipes2.append([tempX1, tempY1, tempX2, tempY2])
        #             pipes4.append([tempX1, tempY1, tempX2, tempY2])
        #             toAdd = 0            
                    
        #            ## if not ( ((xLb-border) <= tempX2 <= (xHb + border)) and ((yLb-border) <= tempY2 <= (yHb + border)) ): ## no arrow at second point              

        #             if (wT>hT): #horizontal
        #                 lineT = hT

        #             elif (hT>wT):
        #                 lineT = wT                            

        #             #print("arrow 2 added")
        #             boxesPx1 = tempX2 - int(3*lineT)
        #             boxesPy1 = tempY2 - int(3*lineT)
        #             boxesPipes.append([boxesPx1, boxesPy1, (boxesPx1 +(lineT*5)), (boxesPy1+(lineT*5))])
        #             cv2.rectangle(im7, (boxesPx1, boxesPy1), (boxesPx1 +(lineT*5), boxesPy1+(lineT*5)), (70, 150, 255), 2)                     
                    
        #         if ((xLb-xBorder) <= tempX2 <= (xLb+xBorder)) and ((yLb-yBorder) <= tempY2 <= (yLb+yBorder)) and (toAdd == 1):                    
        #         #if ((xLb-border) <= tempX2 <= (xHb + border)) and ((yLb-border) <= tempY2 <= (yHb + border)) and (toAdd == 1):
        #             #print("line next to arrow 2")                           
        #             cv2.rectangle(im7, (tempX1, tempY1), (tempX2, tempY2), (255, 255, 0), 5)
        #             cv2.rectangle(im8, (tempX1, tempY1), (tempX2, tempY2), (255, 0, 0), -1)
        #             pipes2.append([tempX1, tempY1, tempX2, tempY2])
        #             pipes4.append([tempX1, tempY1, tempX2, tempY2])  
        #             toAdd = 0             
                    
        #             ##if not ( ((xLb-border) <= tempX1 <= (xHb + border)) and ((yLb-border) <= tempY1 <= (yHb + border)) ): ## no arrow at second point              

        #             if (wT>hT): #horizontal
        #                 lineT = hT
        #             elif (hT>wT):
        #                 lineT = wT                            

        #             #print("arrow 1 added")
        #             boxesPx1 = tempX1 - int(2*lineT)
        #             boxesPy1 = tempY1 - int(2*lineT)
        #             boxesPipes.append([boxesPx1, boxesPy1, (boxesPx1 +(lineT*5)), (boxesPy1+(lineT*5)) ])
        #             cv2.rectangle(im7, (boxesPx1, boxesPy1), (boxesPx1 +(lineT*5), boxesPy1+(lineT*5)), (70, 150, 255), 2)                        
                    
                    
    print("length of pipes2 before boxes", len(pipes2))                     
    print("length of pipes4", len(pipes4))       

    
    pipes2C = []

    lineSizeFind = []

    for a in linesToMeasure:
        if (a[3]-a[1] >2):
            lineSizeFind.append(a[3]-a[1])
    
    from collections import Counter

    print('lineSizeFind', Counter(lineSizeFind))

    if (len(lineSizeFind) != 0):
        #print('most common', mode(lineSizeFind))

        try:
            print('most common', mode(lineSizeFind))
            lineSizeDetect = mode(lineSizeFind)
        except StatisticsError:
            print("no common value")
            lineSizeDetect = 8000 ## 

    else:
        lineSizeDetect = 8000 ## 
    

    for k in range(0, len(pipes)):
            #l=0
            #toAdd=1
            #borderP = 20
            #print(pipes[i][1])
 
            
            y2T = pipes[k][3] + yC
            y1T = pipes[k][1] + yC
            x1T = pipes[k][0] + xC
            x2T = pipes[k][2] + xC
            
            #cv2.rectangle(im7, (x1T, y1T), (x2T, y2T), (0, 0, 255), 10) 
            
            
            
            listT = [x1T, y1T, x2T, y2T] 
            
            if listT not in pipes2:
                
                for i in range(0, len(boxesPipes)):
                
                    if(boxesPipes[i][0]<=x1T<=boxesPipes[i][2]) and (boxesPipes[i][1]<=y1T<=boxesPipes[i][3]):
                        
                        pipes2.append([x1T, y1T, x2T, y2T])
                        cv2.rectangle(im7, (x1T, y1T), (x2T, y2T), (255, 0, 0), 10)  
                        cv2.rectangle(im8, (x1T, y1T), (x2T, y2T), (255, 0, 0), -1)
                        
                    elif(boxesPipes[i][0]<=x2T<=boxesPipes[i][2]) and (boxesPipes[i][1]<=y2T<=boxesPipes[i][3]):
                        
                        pipes2.append([x1T, y1T, x2T, y2T])                
                        cv2.rectangle(im7, (x1T, y1T), (x2T, y2T), (255, 0, 0), 10)  
                        cv2.rectangle(im8, (x1T, y1T), (x2T, y2T), (255, 0, 0), -1)
                        
    print("length of pipes2 after boxes", len(pipes2))    
    
    pipeAdded = 1
   # n = 0
   # p = 0
   # while(p<4):
    a = 10
    #p = p+1
    pipeAdded = 0
   # n = n+1
    
    for k in range(0, len(pipes)):#split lines
            #l=0
            #toAdd=1
            #borderP = 20
            #print(pipes[i][1])
        
        y2T = pipes[k][3] + yC
        y1T = pipes[k][1] + yC
        x1T = pipes[k][0] + xC
        x2T = pipes[k][2] + xC
        
        if(x1T>x2T):
            print(" x co-ord")
        elif(y1T>y2T):
            print(" y co-ord")            
            #cv2.rectangle(im7, (x1T, y1T), (x2T, y2T), (0, 0, 255), 10) 
            
        listT = [x1T, y1T, x2T, y2T] 
        #print('still running')
        
        if listT not in pipes2:
            #print("t")
            
            for i in range(0, len(pipes2)):
                #if (i%500== 0):
                    #print('running', i)
                
                if((pipes2[i][3]-pipes2[i][1])>(pipes2[i][2]-pipes2[i][0])):   
                    #if(pipes2[i][2]-pipes2[i][0]>4):
                    if((x1T-a)<=pipes2[i][0]<=(x1T+a)) and ((x2T-a)<=pipes2[i][2]<=(x2T+a)):
                        if(5<(pipes2[i][1]-y2T)<100) or (5<(pipes2[i][3]-y1T)<100):                        
                            
                            pipes2.append([x1T, y1T, x2T, y2T])                
                            cv2.rectangle(im8, (x1T, y1T), (x2T, y2T), (0, 0, 255), -1)   
                            cv2.rectangle(im7, (x1T, y1T), (x2T, y2T), (0, 0, 255), 10)                     
                            cv2.rectangle(im7, (pipes2[i][0], pipes2[i][1]), (pipes2[i][2], pipes2[i][3]), (0, 255, 0), 4)
                            pipeAdded=1
                            
                if((pipes2[i][2]-pipes2[i][0])>(pipes2[i][3]-pipes2[i][1])):             
                    #if(pipes2[i][3]-pipes2[i][1]>4):
                    if((y1T-a)<=pipes2[i][1]<=(y1T+a)) and ((y2T-a)<=pipes2[i][3]<=(y2T+a)):
                        if(5<(x1T-pipes2[i][2])<100) or (5<(pipes2[i][0]-x2T)<100):   
                            
                            pipes2.append([x1T, y1T, x2T, y2T])                
                            cv2.rectangle(im8, (x1T, y1T), (x2T, y2T), (0, 0, 255), -1)   
                            cv2.rectangle(im7, (x1T, y1T), (x2T, y2T), (0, 0, 255), 10)               
                            cv2.rectangle(im7, (pipes2[i][0], pipes2[i][1]), (pipes2[i][2], pipes2[i][3]), (0, 255, 0), 4)     
                            pipeAdded=1
                        
    print("length of pipes2 after split lines",  len(pipes2))           
    pipe2 = []
    pipeAdded = 1
   # n = 0
   # p = 0
   # while(p<4):
    a = 10
    #p = p+1
    pipeAdded = 0
    #n = n+1
    
    for k in range(0, len(pipes)):#split lines
            #l=0
            #toAdd=1
            #borderP = 20
            #print(pipes[i][1])
        
        y2T = pipes[k][3] + yC
        y1T = pipes[k][1] + yC
        x1T = pipes[k][0] + xC
        x2T = pipes[k][2] + xC
        
        if(x1T>x2T):
            print(" x co-ord")
        elif(y1T>y2T):
            print(" y co-ord")            
            #cv2.rectangle(im7, (x1T, y1T), (x2T, y2T), (0, 0, 255), 10) 
            
        listT = [x1T, y1T, x2T, y2T] 
        
        if listT not in pipes2:
            #print("t")
            
            for i in range(0, len(pipes2)):
                
                if((pipes2[i][3]-pipes2[i][1])>(pipes2[i][2]-pipes2[i][0])):   
                    if(pipes2[i][2]-pipes2[i][0]>4):
                        if((x1T-a)<=pipes2[i][0]<=(x1T+a)) and ((x2T-a)<=pipes2[i][2]<=(x2T+a)):
                            if(5<(pipes2[i][1]-y2T)<100) or (5<(pipes2[i][3]-y1T)<100):                        
                                
                                pipes2.append([x1T, y1T, x2T, y2T])                
                                cv2.rectangle(im8, (x1T, y1T), (x2T, y2T), (0, 0, 255), -1)   
                                cv2.rectangle(im7, (x1T, y1T), (x2T, y2T), (0, 0, 255), 10)                     
                                cv2.rectangle(im7, (pipes2[i][0], pipes2[i][1]), (pipes2[i][2], pipes2[i][3]), (0, 255, 0), 4)
                                pipeAdded=1
                            
                if((pipes2[i][2]-pipes2[i][0])>(pipes2[i][3]-pipes2[i][1])):             
                    if(pipes2[i][3]-pipes2[i][1]>4):
                        if((y1T-a)<=pipes2[i][1]<=(y1T+a)) and ((y2T-a)<=pipes2[i][3]<=(y2T+a)):
                            if(5<(x1T-pipes2[i][2])<100) or (5<(pipes2[i][0]-x2T)<100):   
                                
                                pipes2.append([x1T, y1T, x2T, y2T])                
                                cv2.rectangle(im8, (x1T, y1T), (x2T, y2T), (0, 0, 255), -1)   
                                cv2.rectangle(im7, (x1T, y1T), (x2T, y2T), (0, 0, 255), 10)               
                                cv2.rectangle(im7, (pipes2[i][0], pipes2[i][1]), (pipes2[i][2], pipes2[i][3]), (0, 255, 0), 4)     
                                pipeAdded=1
                        
    print("length of pipes2 after split lines", len(pipes2))           
    pipe2 = []
        
        ##remove thin lines


    ## join up the linesToMeasure 

    joinedLines =[]
    change = 0
    linesToMeasure2 = linesToMeasure.copy()
    newLines = []
    linesToMeasureJoined = []
    totalMatchingLines = []

    from random import randrange


    counter = 1

    linesUsed = []


    for line in linesToMeasure:

        line_midY = int(line[1] + ((line[3] - line[1])/2) )

        if line not in linesUsed: 

            reset = 0

            linesToJoin = []
            linesToJoin.append(line)


            for line2 in linesToMeasure:

               # if line != line2 : 

                line2Added = 0

                line2_midY = int(line2[1] + ( (line2[3] - line2[1])/2) ) 

                if ( abs(line_midY - line2_midY) < thicknessMargin ) : ##assume its the same line and draw with the same colour as line1


                    for l in linesToJoin:
                          

                        if ( ( (l[0]-line2[2]) > 0 ) and  ( (l[0]-line2[2]) < spacesMargin ) )  or ( ( (line2[0]-l[2]) > 0) and ( (line2[0]-l[2]) < spacesMargin ) ) :      ## assume its meant to be the same line by checking for the spacesMargin

                        #if ( spacesMargin > (line[0]-line2[2]) > 0 ) or ( spacesMargin > (line[2]-line2[0]) > 0):      ## assume its meant to be the same line by checking for the spacesMargin

                            if ( line2Added == 0 ) :

                                #print('close match',line_midY, line2_midY, l[1], line2[1], l[3], line2[3], 'x space ', (l[0]-line2[2]), (l[2]-line2[0]) )

                                #cv2.rectangle(im200,(line2[0],line2[1]),(line2[2],line2[3]),colour,5)

                                linesToJoin.append(line2)

                                if ( reset == 0 ):
                                    linesUsed.append(line)
                                    reset = 1

                                linesUsed.append(line2)

                                line2Added = 1

            colour = ( randrange(245),randrange(245), randrange(245) ) 
            
            for l in linesToJoin:


                if ((l[2]-l[0]) == lineSizeDetect) or ((l[2]-l[0]) == (lineSizeDetect + 1)) or ((l[3]-l[1]) == lineSizeDetect) or ((l[3]-l[1]) == (lineSizeDetect+1)) :

                    cv2.rectangle(im200,(l[0],l[1]),(l[2],l[3]),colour,5)



                    # else: # if on the same y but not the same pipe, use different colour

                    #     print('')

                        # new_colour = ( randrange(245),randrange(245), randrange(245) )

                        # cv2.rectangle(im200,(line2[0],line2[1]),(line2[2],line2[3]),new_colour,5)

            ## check if linesToJoin has code of interest 
            for ((startX, startY, endX, endY), text) in results:
                text = "".join([c for c in text]).strip()  
                for item in text_i: # for each code of interest that is in text_i, 
                    if item in text: #if the detected text is in the list of codes of interest 

                        lowestXJoinedLine = math.inf
                        highestXJoinedLine = 0

                        lowestYJoinedLine = math.inf
                        highestYJoinedLine = 0

                        for l in linesToJoin:


                            if (l[0] < lowestXJoinedLine):
                                lowestXJoinedLine = l[0]
                            if (l[2] > highestXJoinedLine):
                                highestXJoinedLine = l[2]
                            if (l[1] < lowestYJoinedLine):
                                lowestYJoinedLine = l[1]
                            if (l[3] > highestYJoinedLine):
                                highestYJoinedLine = l[3]

                        if (lowestXJoinedLine < startX ) and (highestXJoinedLine > endX): ##then the code of interest is in the line, draw on the diagram in red
                            if ( ( lowestYJoinedLine - margin2) < (startY) ) and ( ( highestYJoinedLine + margin2) > endY ): 

                                if ((l[2]-l[0]) == lineSizeDetect) or ((l[2]-l[0]) == (lineSizeDetect+1)) or ((l[3]-l[1]) == lineSizeDetect) or ((l[3]-l[1]) == (lineSizeDetect+1)) :

                                    cv2.rectangle(im300,(startX, startY),(endX, endY),(0,255,0),3)
                                    for l in linesToJoin:

                                        if ((l[2]-l[0]) == lineSizeDetect) or ((l[2]-l[0]) == (lineSizeDetect+1)) or ((l[3]-l[1]) == lineSizeDetect) or ((l[3]-l[1]) == (lineSizeDetect+1)) :


                                            cv2.rectangle(im300,(l[0],l[1]),(l[2],l[3]),(0,0,255),-1)

                                     ## draw combined line in thin blue line
                                        if ((highestXJoinedLine-lowestXJoinedLine) == lineSizeDetect) or ((highestXJoinedLine-lowestXJoinedLine) == (lineSizeDetect+1)) or ((highestYJoinedLine-lowestYJoinedLine) == lineSizeDetect) or ((highestYJoinedLine-lowestYJoinedLine) == (lineSizeDetect+1)) :

                                            cv2.rectangle(im300,(lowestXJoinedLine,lowestYJoinedLine),(highestXJoinedLine,highestYJoinedLine),(255,0,0), 2)
                                            pipeInterest.append([lowestXJoinedLine,lowestYJoinedLine,highestXJoinedLine,highestYJoinedLine])

    for p in range(0, len(pipeInterest)): ##remove the adjustment for whole drawing
 
        
        pipeInterest[p][0] = pipeInterest[p][0] - xC        
        pipeInterest[p][1] = pipeInterest[p][1] - yC
        pipeInterest[p][2] = pipeInterest[p][2] - xC
        pipeInterest[p][3] = pipeInterest[p][3] - yC      

        cv2.rectangle(im4, (pipeInterest[p][0], pipeInterest[p][1]), (pipeInterest[p][2], pipeInterest[p][3]), (255, 0, 0), -1)



    
                                
    for p in range(0, len(pipes2)): ##remove the adjustment for whole drawing
 
        cv2.rectangle(im4, (pipes2[p][0], pipes2[p][1]), (pipes2[p][2], pipes2[p][3]), (255, 255, 0), 5)
        
        pipes2[p][0] = pipes2[p][0] - xC        
        pipes2[p][1] = pipes2[p][1] - yC
        pipes2[p][2] = pipes2[p][2] - xC
        pipes2[p][3] = pipes2[p][3] - yC        

    for k in range(0, len(pipes2)):

        minT=3
        maxT = 15
        
        y2T = pipes2[k][3]
        y1T = pipes2[k][1]
        x1T = pipes2[k][0]
        x2T = pipes2[k][2]
        
        if ((x2T-x1T)>minT) and ((y2T-y1T)>minT):     
            if (x2T-x1T)>(y2T-y1T):
                if (y2T-y1T)<maxT:
                    pipe2.append([x1T, y1T, x2T, y2T])                
            if (y2T-y1T)>(x2T-x1T):
                if (x2T-x1T)<maxT:
                    pipe2.append([x1T, y1T, x2T, y2T])              
            
        
    print("no h, v, total", h, v, h+v)                       
    name = currentPID[:-4]+"pipe.png"
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], name), im)                
    name2 = currentPID[:-4]+"pipe_2.png"
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], name2), im2)     
    name3 = currentPID[:-4]+"pipe_3.png"
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], name3), im3)     
    name4 = currentPID[:-4]+"pipe_4.png"
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], name4), im4)   
    
    name6 = currentPID[:-4]+"pipe_6.png"
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], name6), im6) 
    
    name7 = currentPID[:-4]+"pipe_7.png"
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], name7), im7) 

    name8 = currentPID[:-4]+"pipe_8.png"
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], name8), im8) 

    name100 = currentPID[:-4]+"pipe_interest.png"
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], name100), im100) 
    
    name200 = currentPID[:-4]+"pipe_interest_joined.png"
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], name200), im200) 


    name300 = currentPID[:-4]+"pipe_interest_only.png"
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], name300), im300) 

    print("end length pipes", len(pipes))                        
    print("length pipes2", len(pipes2))   
    print("no h, v", h, v)      
    
    
#    return pipes2


    #return pipe2
    return pipeInterest








## detectPipelines function ###################################################################################################################################
## function to detect pipelines in diagram ################################################################################################################################### 
def detectPipelines():
       
    global pipes
    global dashes
    global xC
    global yC
    global LABELS
 
    lineType = []   
    no = 0
    no1 = 0
    minLineSize = 10   
    p=0
    itemN = 0
    no2 = 0
    lineSize = 15
    
    filenameToRead = currentPID[:-4] + "_detection.png"
    
    #imageOverlap = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameToRead))
    #imageOver = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameToRead))    
    image = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameToRead))##read image
    imTest2 = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameToRead))##read image
    
    imIH, imIW = image.shape[:2]
    
    ##lineSizesBoth = findLineSizes()
    
    #detectArrows()
    
    #removeArrows()

    filenameImageA2 = currentPID[:-4] + "_A2.png" ##pre-processed image
    inputImage = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameImageA2), 0)   
    
    imH, imW = inputImage.shape[:2]

    if (isCropped == 0):
        inputImage = inputImage[yC:(yC+ hC), xC:(xC + wC)]    
        
    img = inputImage    
    
    bin_thresh = 180
    
    threshold, imgThresholded = cv2.threshold(img, bin_thresh, 255, 1, cv2.THRESH_BINARY_INV) 
    horizontal_lines_size = lineSize
    #horizontal_lines_size = int(max(lineSizesBoth)*1.5) 
    
    horizontalStructure_lines = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_lines_size,1))  
    horizontalLinesImg = cv2.erode(imgThresholded, horizontalStructure_lines, iterations=1)           
    horizontalLinesImg = cv2.dilate(horizontalLinesImg, horizontalStructure_lines,iterations=1)       

    f4 = currentPID[:-4] + "_horizontal.png"
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], f4), horizontalLinesImg) 
    
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(horizontalLinesImg, connectivity=4)  
    sizes = stats[:, -1]
    #print("horizontal lines", nb_components)
    #image2 = cv2.imread(imageName)
    fi = currentPID[:-4] + "_A2.png"
    image2 = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], fi))
    
    
    f5 = currentPID    
    image5 = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], f5))    
    
    imTest = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], f4), 1)
    
    max_label = 1
    max_size = sizes[1]
    
    min_size = 50000
    
    min_length = 50000
    max_length = 1
    
    hHeight = []
    yTL=[]
    hTmax = 0
    
    
    #global dashesSize
    #dashesSize = int(0.0005*imW)
    dashesSize = 30
    print("dashes size is ", dashesSize)
    noC = 0
    global arrows
    for i in range(1, nb_components):
            
            assigned = 0 
        
            xT = stats[i,0]
            yT = stats[i,1]
            wT = stats[i,2]
            hT = stats[i,3]    
            
            imS = horizontalLinesImg[yT:yT+hT, xT:xT+wT]

            fS = "subC.png"
            #cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], fS), imS)            
            
            ##image from cc box
            ##check if 5 c within box as before 
            
            if hT>hTmax:
                hTmax = hT
#            a = randint(0,255)
#            b = randint(0,255)
#            c = randint(0,255)
            
            hHeight.append(hT)
            yTL.append(yT)
            
            if sizes[i] < min_size:
                min_size = sizes[i]        

            
            if (wT>(dashesSize) and (assigned==0) ):
        
                cv2.rectangle(image2, (xT+xC, yT+yC), (xT + wT+xC, yT + hT+yC), (255,0,0), -1)  
                pipes.append([xT, yT, (xT + wT), (yT + hT)])
                
            if (wT<(dashesSize) and (assigned==0) ):
                cv2.rectangle(image2, (xT+xC, yT+yC), (xT + wT+xC, yT + hT+yC), (0,255,0), -1)  
                dashes.append([xT, yT, (xT + wT), (yT + hT)])   
            
             
            if wT < min_length:
            #max_label = i
                min_length = wT
            
            if wT > max_length:
                max_length = wT
    
    f5 = "c_test.png"            
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], f5), imTest)
    f7 = "c_test2.png"            
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], f7), imTest2)    
    print("horizontal lines size, max height", horizontal_lines_size, hTmax)
    
    #vertical_lines_size = max(lineSizesH) + 1
    #vertical_lines_size = int(max(lineSizesBoth)*1.5)    
    vertical_lines_size = lineSize
    verticalStructure_lines = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_lines_size))
    verticalLinesImg = cv2.erode(imgThresholded, verticalStructure_lines, iterations=1)
    verticalLinesImg = cv2.dilate(verticalLinesImg, verticalStructure_lines,iterations=1)

    f5 = currentPID[:-4] + "_vertical.png"
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], f5), verticalLinesImg)
    
    nb_components2, output2, stats2, centroids2 = cv2.connectedComponentsWithStats(verticalLinesImg, connectivity=4)
    sizes2 = stats2[:, -1]
    print("vertical lines", nb_components2)
    linesFoundV = []
    wTmax = 0
    
    for i in range(1, nb_components2):
        
            xT = stats2[i,0]
            yT = stats2[i,1]
            wT = stats2[i,2]
            hT = stats2[i,3]    
            
            if wT>wTmax:
                wTmax = wT
                

            
            if (hT>(dashesSize)):
        
                cv2.rectangle(image2, (xT+xC, yT+yC), (xT + wT+xC, yT + hT+yC), (0,0,255), -1) 
                pipes.append([xT, yT, (xT + wT), (yT + hT)])

            if (hT<(dashesSize)):
        
                cv2.rectangle(image2, (xT+xC, yT+yC), (xT + wT+xC, yT + hT +yC), (255,0,255), -1) 
                dashes.append([xT, yT, (xT + wT), (yT + hT)])      

    print("vertical lines size, max width", vertical_lines_size, wTmax)
     
         
    #cv2.imwrite("wholePID.png", image2)
     
    print("length of pipes after detectPipelines", len(pipes))        
    print("length of dashes after detectPipelines", len(dashes))       
    
    global pipesDetected
    pipesDetected = pipes.copy()
    
    global pipelines
    pipelines = []
    global nameForCSVFile       
    global itemNo
    print("itemNo before detect pipelines:", itemNo)
    global scanAreaX
    
    
    for pipe in range(0, len(pipes)):
        cv2.rectangle(image5, (pipes[pipe][0]+xC, pipes[pipe][1]+yC), (pipes[pipe][2]+xC, pipes[pipe][3]+yC), (0,0,255), -1) 
    
    
    print("before symbol check no of pipelines: ", len(pipes))
    pipes = pipesCheck()
    print("after symbol check no of pipelines: ", len(pipes))    
    
    for pipe in range(0, len(pipes)):
        cv2.rectangle(image5, (pipes[pipe][0]+xC, pipes[pipe][1]+yC), (pipes[pipe][2]+xC, pipes[pipe][3]+yC), (255,0,0), -1) 

    print("before text check no of pipelines: ", len(pipes))
    pipes = pipesCheckText2(pipes)
    print("after text check no of pipelines: ", len(pipes))  

    for pipe in range(0, len(pipes)):
        cv2.rectangle(image5, (pipes[pipe][0]+xC, pipes[pipe][1]+yC), (pipes[pipe][2]+xC, pipes[pipe][3]+yC), (0,255,0), -1)     

    f = currentPID[:-4]+ "lines.png"
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], f), image2)   
    
    f = currentPID[:-4]+"lines_5.png"
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], f), image5)   
    
    print("no of dashes before text check:", len(dashes))    
    dashes = dashesCheckText(dashes)
    print("no of dashes after text check:", len(dashes))  
    
    
    print("after dashes check no of pipelines: ", len(pipes))     

    # pipesSet = []
    # pipesSet = set(pipes)
    # pipes = []
    # pipes = list(pipesSet)

    pipesDuplicates = []
    pipesDuplicates = pipes.copy()

    pipes = []

    for p in pipesDuplicates:
        if p not in pipes:
            pipes.append(p)

    print("unique no of pipelines: ", len(pipes))  
    
    with open(os.path.join(app.config['UPLOAD_FOLDER'], nameForCSVFile), 'a', newline = '') as outfile, open(os.path.join(app.config['UPLOAD_FOLDER'], nameForCSVFilePipelines), 'a', newline = '') as outfilePipelines:
        csvWriter = csv.writer(outfile, delimiter = ',')
        csvWriter.writerow(['detected Pipelines:'])        
        csvWriter.writerow(['diagramNo', 'item', 'itemNo', 'x1', 'y1', 'x2', 'y2', 'length m', 'dimension', 'code'])

        csvWriterPipelines = csv.writer(outfilePipelines, delimiter = ',')
        csvWriterPipelines.writerow(['detected Pipelines:'])        
        csvWriterPipelines.writerow(['diagramNo', 'item', 'itemNo', 'x1', 'y1', 'x2', 'y2'])            
            
    pidNo = currentPID[:-4]
    
    
    
    
    with open(os.path.join(app.config['UPLOAD_FOLDER'], nameForCSVFile), 'a', newline = '') as outfile, open(os.path.join(app.config['UPLOAD_FOLDER'], nameForCSVFilePipelines), 'a', newline = '') as outfilePipelines:                  
        csvWriter = csv.writer(outfile, delimiter = ',')  
        csvWriterPipelines = csv.writer(outfilePipelines, delimiter = ',')          
        print("length of pipelines for writing to csv file is:", len(pipes))
        print("test, origW, origH", origW, origH)
        for n in range (0, len(pipes)):
            
            length = 0
            scale = 100
            pipes[n][0] =     pipes[n][0] + xC            
            pipes[n][1] =     pipes[n][1] + yC                 
            pipes[n][2] =     pipes[n][2] + xC                 
            pipes[n][3] =     pipes[n][3] + yC   

            if (pipes[n][3] - pipes[n][1])> (pipes[n][2] - pipes[n][0]):
                length = pipes[n][3] - pipes[n][1]
                length = ((((length/origW)*594)*scale)/1000)   #####

            else:
                length = pipes[n][2] - pipes[n][0]
                length = ((((length/origH)*841)*scale)/1000)   #####
         
            itemNo = itemNo + 1

            assignedDimension, assignedCode = findDimension(pipes[n][0], pipes[n][1], pipes[n][2], pipes[n][3], 200)

                        
            csvWriter.writerow([pidNo, 'pipe', itemNo, pipes[n][0], pipes[n][1], pipes[n][2], pipes[n][3], length, assignedDimension, assignedCode ])    
            csvWriterPipelines.writerow([pidNo, 'pipe', itemNo, pipes[n][0], pipes[n][1], pipes[n][2], pipes[n][3]])
            
            
        
        
        
        ######csvWriterPipelines.writerow(['pidNo', 'Labelled Class', 'x', 'y', 'width', 'height', 'Predicted Class', 'itemNumber', 'x', 'y', 'width', 'height', 'IOU', 'matching class'])
        
    for n in range (0, len(dashes)):
        
        
        dashes[n][0] =     dashes[n][0] + xC            
        dashes[n][1] =     dashes[n][1] + yC                 
        dashes[n][2] =     dashes[n][2] + xC                 
        dashes[n][3] =     dashes[n][3] + yC    
    
  
    
    filenameProcessed = currentPID[:-4] + "_detection.png"
    
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], filenameProcessed), image) ##**save the patch with the detections here #save image

    timeValue = str(int(time.time()))
    
    global filenamePIDTimeS
    filenamePIDTimeS = currentPID[:-4] + "_" + timeValue + "_detection.png"#save image with time
    
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], filenamePIDTimeS), image)
    print("item no after detect pipelines:", itemNo)
    print("after detectPipelines no of pipelines: ", len(pipes))   
    
    global totalItems
    totalItems = itemNo



## createItemList function ###################################################################################################################################
## function to create a list of found components  ################################################################################################################################### 
def createItemList():
    
    global boxesPID
    global COLORS
    global LABELS
    global detectNo
    global symbolCount
    global itemList
    global results
    global pipes
    
    symbolCount = 0
    print("item count before symbols:", symbolCount)      
    
    for k in range(len(boxesPID)):
        
        symbolCount = symbolCount + 1

        itemList.append([symbolCount, (boxesPID[k][0]), (boxesPID[k][1]), (boxesPID[k][0] + boxesPID[k][2]), (boxesPID[k][1] + boxesPID[k][3]), "symbol", LABELS[boxesPID[k][4]]])
    
    
    print("item count after symbols:", symbolCount)
    
    for ((startX, startY, endX, endY), text) in results:       
        symbolCount = symbolCount +1

        itemList.append([symbolCount, startX, startY, endX, endY, "text", text])
             
    
    for i in range(0, len(pipes)): 
             
           symbolCount = symbolCount + 1       
           itemList.append([symbolCount, pipes[i][0], pipes[i][1], pipes[i][2], pipes[i][3], "pipeline", "pipeline"]) 
           
    print("total item count:", symbolCount)          

## drawSymbols function ###################################################################################################################################
## function to draw onto the imageInput, boxes round the detected symbols  ################################################################################################################################### 
def drawSymbols(imageInput):
    
        print("drawSymbols")
    
        global boxesPID
        global COLORS
        global LABELS
        global detectNo
        global symbolCount
        global itemList
        global boxesPID_updated

        global high_pred
        global low_pred

       # symbolCount = 0
        detectNo = 0
        q=0
        print("symbol count before symbols drawn on diagram:", detectNo) 

        print('length of boxesPID:, length of boxesPID_updated: ', len(boxesPID), len(boxesPID_updated))
        
        for k in range(len(boxesPID)):

            if(boxesPID[k][5] >= conf_al_threshold):
                     colorB = (210, 0, 150)
                     high_pred.append([ boxesPID[k][0], boxesPID[k][1], boxesPID[k][2], boxesPID[k][3], boxesPID[k][4], boxesPID[k][5], boxesPID[k][6] ] )

            else: 

                     colorB = (30, 100, 255)
                     low_pred.append( [ boxesPID[k][0], boxesPID[k][1], boxesPID[k][2], boxesPID[k][3], boxesPID[k][4], boxesPID[k][5], boxesPID[k][6] ] )

 
            #colorB = [int(c) for c in COLORS[boxesPID[k][4]]]
            
            #    if (LABELS[boxesPID[k][4]] == "Sensor"):
            #         colorB = (0, 0, 255)


            # if(boxesPID[k][4]==29):
            #         colorB = (255, 0, 255)
                    
            #else:
                #colorB = (255, 0, 255)

                    
            cv2.rectangle(imageInput, (boxesPID[k][0], boxesPID[k][1]), (boxesPID[k][0] + boxesPID[k][2], boxesPID[k][1] + boxesPID[k][3]), colorB, 2)
            #symbolCount = symbolCount + 1
            detectNo = detectNo + 1
            #text = "{}: {:.4f}".format(LABELS[boxesPID[k][4]], boxesPID[k][5]) #labels with confidence values
            
            #if (boxesPID[k][4]<=28):
            #text = "{}".format(LABELS[boxesPID[k][4]]) #labels without confidence values
            text = "{} {:.1f}".format(LABELS[boxesPID[k][4]], boxesPID[k][5]) #labels with iou and confidence values
                


            imH, imW, imD = imageInput.shape
            
            if((boxesPID[k][1] + boxesPID[k][3] + 30) <=imH):
                yT = boxesPID[k][1] + boxesPID[k][3] + 30
            else:
                yT = imH
                
            cv2.putText(imageInput, text, (boxesPID[k][0], yT), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colorB, 2)
            text = str(detectNo)

            if (detectNo != boxesPID[k][6]):
                print('mismatch numbers', detectNo, boxesPID[k][6] )
          
            if((boxesPID[k][1] + boxesPID[k][3] + 50) <= imH):
                yT = boxesPID[k][1] + boxesPID[k][3] + 50
            else:
                yT = imH
                         
            drawNumber(imageInput, text, boxesPID[k][0], yT)   
        
        print('number of symbols drawn on diagram: ', detectNo)    
        print(' total h l ', len(boxesPID), len(high_pred), len(low_pred))           

#### detailedSymbolsData function ###############################################################################################################################
################################################################################################################################################################################

def detailedSymbolsData():

    global nameForCSVFile
    nameForCSVFileTest = nameForCSVFile[:-4] + '_symbol_dimensions.csv'
    with open(os.path.join(app.config['UPLOAD_FOLDER'], nameForCSVFileTest), 'a', newline = '') as outfile:    
        csvWriter = csv.writer(outfile, delimiter = ',')

        csvWriter.writerow(['Predicted Class', 'item number', 'x', 'y', 'width', 'height', 'dimension'])

        for k in range(len(boxesPID)): 
            classTemp = LABELS[boxesPID[k][4]]                
            b = findDimensionS(boxesPID[k][0], boxesPID[k][1], (boxesPID[k][0] + boxesPID[k][2]), (boxesPID[k][1] + boxesPID[k][3]) )
            csvWriter.writerow([classTemp, boxesPID[k][6], boxesPID[k][0], boxesPID[k][1], boxesPID[k][2], boxesPID[k][3], b])

    return a


## detectSymbols function ###################################################################################################################################
## function to detect symbols in drawing ################################################################################################################################### 
def detectSymbols():
    
    print("diagram name is: ", currentPID)
    
    filename = currentPID
    filenameToReadOutputS = currentPID[:-4] + "_detection.png"
    imageOut = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameToReadOutputS))    
    
    filenameToRead = currentPID  ##reads the uploaded file   
    imageWhole = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameToRead))


    bin_thresh_ = 0
    preprocess = 90

    if (preprocess == 1):
        print('preprocessing')
        threshold, imageWhole = cv2.threshold(imageWhole, bin_thresh_, 255, cv2.THRESH_BINARY)


        #f11 = imageFilename[:-4] + '_bitwise_not_original_c.png'
        bitwise_not_original = cv2.bitwise_not(imageWhole)
        #cv2.imwrite(f11, bitwise_not_original)



        kernelSize = (3, 3)

        # loop over the kernels sizes
        #for kernelSize in kernelSizes:
            # construct a rectangular kernel from the current size and then
            # apply an "opening" operation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
        opening = cv2.morphologyEx(bitwise_not_original, cv2.MORPH_OPEN, kernel)

        #f2 = imageFilename[:-4] + str(kernelSize[0]) + '_.png'

        #cv2.imwrite(f2, opening)



        #f11 = imageFilename[:-4] +  str(kernelSize[0]) + '_bitwise_not_output_d.png'
        bitwise_not_original_d = cv2.bitwise_not(opening)
        #cv2.imwrite(f11, bitwise_not_original_d)

        imageWhole = bitwise_not_original_d

        if (isCropped == 0):

            

            print(xC, yC, hC, wC, isCropped)

            imageMask = 255*np.ones_like(imageWhole)
            imageMask[yC:yC+hC, xC:xC+wC] = imageWhole[yC:yC+hC, xC:xC+wC]

            imageWhole = imageMask

        fi = filenameToRead[:-4] + '_' + str(bin_thresh_) + '_pp.png'
        cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], fi), imageWhole) #

    if (preprocess == 2):  ##increase the contrast using convertScaleAbs

        alpha = 3.0 # contrast, 1.0-3.0 
        beta = 0.0 # brightness

        imgContrasted = cv2.convertScaleAbs(imageWhole, alpha=alpha, beta=beta)

        imageWhole = imgContrasted

        fi = filenameToRead[:-4] + '_contrast_pp.png'
        cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], fi), imageWhole) #

    if (preprocess == 3):  ##increase the contrast using convertScaleAbs

        src = cv2.cvtColor(imageWhole, cv2.COLOR_BGR2GRAY)
        dst = cv2.equalizeHist(src)


        imageWhole = dst

        fi = filenameToRead[:-4] + '_equalise_hist_image.png'
        cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], fi), imageWhole) #

    else:
        print(' ')
        fi = filenameToRead[:-4] + '_no_pp_applied.png'
        cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], fi), imageWhole) #

    imageWholeGrid = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameToRead))

    height, width = imageWhole.shape[:2]
    
    imageA1 = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameToRead)) 
    imageA2 = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameToRead)) 

    pidNo = filename[:-4]
    global boxesPID
    global boxesFull
    global confidencesFull
    global classIDsFull
    boxesPID = [] #list for boxes in the full image storing x,y,w,h,classID,confidence


    #pathToYoloFolder = "yolo-symbol" ############
    pathToYoloFolder = "yolo-symbolv2" ############
    confidenceMin = conf_threshold
    thresholdNMS = 0.3


    labelsPath = os.path.sep.join([pathToYoloFolder, names_file])############
    global LABELS
    LABELS = open(labelsPath).read().strip().split("\n")
    
    np.random.seed(42)
    global COLORS
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
        dtype="uint8")
    #print (*COLORS, sep = '\n')

    weightsPath = os.path.sep.join([pathToYoloFolder, weights_file])############
    configPath = os.path.sep.join([pathToYoloFolder, cfg_file])############
    
    global itemNo
    global scanAreaY
    global scanAreaX
    global highestXS
    global lowestXS
    global highestYX
    global lowestYX
    global symbolNumber

    image_patches = 1

    if (image_patches == 1):

        cfg_size = 2400

        imageC = imageWhole
        imageShowGrid = imageWhole
        
        height_im, width_im = imageC.shape[:2]

        step = 4800
        overlap = 400

        height, width = imageC.shape[:2]
        print(height, width)

        r_list_2 = []
        c_list_2 = []

        windowsize_r = step #height -1 for rounding error to allow grid of 4 x 6
        windowsize_c = step #width -1 for rounding error to allow grid of 4 x 6

        windowsize_step = step


        for r in range(0,imageC.shape[0] - windowsize_r, (windowsize_step-overlap)):###   y 
            r_list_2.append(r)

        if (height-(r_list_2[-1])  >= 1):
            r_list_2.append(height-windowsize_r) 


        for c in range(0,imageC.shape[1] - windowsize_c, (windowsize_step-overlap)):###   x
            c_list_2.append(c)

        if (width-(c_list_2[-1]) >= 1):
            c_list_2.append(width-windowsize_c) 


        print(r_list_2, (r_list_2[-1] + step) )
        print(c_list_2, (c_list_2[-1] + step) )


        
        # windowsize_r_ = int(height_im/4) - 1 #height -1 for rounding error to allow grid of 4 x 6
        # windowsize_c_ = int(width_im/6) - 1 #width -1 for rounding error to allow grid of 4 x 6


        # windowsize_r_ = 4800 #height -1 for rounding error to allow grid of 4 x 6
        # windowsize_c_ = 4800 #width -1 for rounding error to allow grid of 4 x 6
        
        patchNumber = 0
        patchesTotal = 0

        print('start symbols across all patches', len(boxesFull), len(confidencesFull), len(classIDsFull) )




        for r in r_list_2 :###   y 
            for c in c_list_2:###   x

                step = step
                
                patchNumber = patchNumber + 1
                image_patch_name = currentPID[:-4] + "_P" + str(patchNumber) + ".png"
                image_patch1 = imageC[r:r+step,c:c+step]    ### y first

                #cv2.imwrite('result/'+image_patch_name, image_patch1)
                cv2.rectangle(imageShowGrid, (c, r), (c+windowsize_c, r+windowsize_r), (0,0,255), 5)

                #print(r,c)
                #numPatches = numPatches + 1
    

        # for r in range(0,imageC.shape[0] - windowsize_r_, windowsize_r_):###y
        #     for c in range(0,imageC.shape[1] - windowsize_c_, windowsize_c_):###x
                
        #         patchNumber = patchNumber + 1
        #         image_patch_name = currentPID[:-4] + "_P" + str(patchNumber) + ".png"
        #         image_patch1 = imageC[r:r+windowsize_r_,c:c+windowsize_c_]

                cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], image_patch_name), image_patch1)
                #cv2.rectangle(imageShowGrid, (c, r), (c+windowsize_c_, r+windowsize_r_), (0,0,255), 1)

                image_patch = image_patch1
                
                print("starting object detection on patch", patchNumber)
                #net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
               
                image = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], image_patch_name))
                (H, W) = image.shape[:2]
                print('image H and W', image_patch_name, H, W)

                net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)                
                ln = net.getLayerNames()
                ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

                blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (cfg_size, cfg_size),  
                    swapRB=True, crop=False)
                net.setInput(blob)
                start = time.time()
                layerOutputs = net.forward(ln)
                end = time.time()
                 
                
                boxes = []
                confidences = []
                classIDs = []
                
                for output in layerOutputs:
                    for detection in output:
                        scores = detection[5:]
                        classID = np.argmax(scores)
                        confidence = scores[classID]
                 
                        if confidence > confidenceMin:
                            box = detection[0:4] * np.array([W, H, W, H])
                            (centerX, centerY, width, height) = box.astype("int")
                 
                            x = int(centerX - (width / 2))
                            y = int(centerY - (height / 2))
                 
                            boxes.append([x, y, int(width), int(height)])
                            confidences.append(float(confidence))
                            classIDs.append(classID)



                idxs1 = cv2.dnn.NMSBoxes(boxes, confidences, confidenceMin, thresholdNMS) ## nms within a patch

                print('symbols in patch', patchNumber, len(idxs1))


                itemNo = 0    
                lowestXS = W
                lowestYS = H
                highestXS = 0
                highestYS = 0
                if len(idxs1) > 0:
                    for i in idxs1.flatten():
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])
                        xFullImage = c + x
                        yFullImage = r + y
                        #xFullImage = x
                        #yFullImage = y
                        (wFullImage, hFullImage) = (w, h)
                        
                        #itemNo = itemNo + 1
                        #itemNumber = itemNo          
                        #boxesPID.append([xFullImage, yFullImage, wFullImage, hFullImage, classIDs[i], confidences[i], itemNumber])

                        boxesFull.append([xFullImage, yFullImage, wFullImage, hFullImage])

                        confidencesFull.append(confidences[i])
                        classIDsFull.append(classIDs[i])
                        

                        #color = [int(c) for c in COLORS[classIDs[i]]]
                        #cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                        #cv2.rectangle(imageA1, (x, y), (x + w, y + h), (0, 0, 255), 4)   
                        #cv2.rectangle(imageA1, (x, y), (x + w, y + h), (255, 255, 255), -1)          
                        # #cv2.rectangle(imageA2, (x, y), (x + w, y + h), (255, 255, 255), -1)
                        
                        # if((x + w)> highestXS):            
                        #     highestXS = x + w   
                        #     symbol = LABELS[classIDs[i]]

                        # if((x)< lowestXS):            
                        #     lowestXS = x   

                        # if((y + h)> highestYS):            
                        #     highestYS = y + h   
                            
                        # if((y)< lowestYS):            
                        #     lowestYS = y  

    ### nms across the whole diagram

    print('symbols across all patches', len(boxesFull), len(confidencesFull), len(classIDsFull) )

    idxs = cv2.dnn.NMSBoxes(boxesFull, confidencesFull, confidenceMin, thresholdNMS)

    print('symbols after whole nms', len(idxs), type(idxs))

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxesFull[i][0], boxesFull[i][1])
            (w, h) = (boxesFull[i][2], boxesFull[i][3])
            xFullImage = x
            yFullImage = y
            #xFullImage = x
            #yFullImage = y
            (wFullImage, hFullImage) = (w, h)
            
            itemNo = itemNo + 1
            itemNumber = itemNo          
            boxesPID.append([xFullImage, yFullImage, wFullImage, hFullImage, classIDsFull[i], confidencesFull[i], itemNumber])

            

            #color = [int(c) for c in COLORS[classIDs[i]]]
            #cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(imageA1, (x, y), (x + w, y + h), (0, 0, 255), 4)   
            cv2.rectangle(imageA1, (x, y), (x + w, y + h), (255, 255, 255), -1)          
            cv2.rectangle(imageA2, (x, y), (x + w, y + h), (255, 255, 255), -1)
            
            if((x + w)> highestXS):            
                highestXS = x + w   
                symbol = LABELS[classIDsFull[i]]

            if((x)< lowestXS):            
                lowestXS = x   

            if((y + h)> highestYS):            
                highestYS = y + h   
                
            if((y)< lowestYS):            
                lowestYS = y  



    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], currentPID[:-4] + '_grid_2.png'), imageShowGrid)


    print("x limits:", lowestXS, highestXS, "y limits:", lowestYS, highestYS)  
  
    imTest = image
    
    cv2.rectangle(imTest, (lowestXS, lowestYS), (highestXS, highestYS), (255, 255, 255), -1)
    
    imTest2 = image
    cv2.rectangle(imTest2, (lowestXS, lowestYS), (highestXS, highestYS), (255, 0, 255), 3)


    
    filenameImageA1 = currentPID[:-4] + "_A1.png"   ##******change the filename here
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], filenameImageA1), imageA1) ##**save the patch with the detections here     
  
    filenameImageA2 = currentPID[:-4] + "_A2.png"   ##******change the filename here
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], filenameImageA2), imageA2) ##**save the patch with the detections here 

    symbolNumber = itemNo
    
#####opencv end
    global annotationsStoreList

    for k in range(len(boxesPID)):
        
        colorB = [int(c) for c in COLORS[boxesPID[k][4]]]
        
        # if (LABELS[boxesPID[k][4]] == "Sensor"):
        #     colorB = (0, 0, 255)
        #     sensorD.append(boxesPID[k][2])
            
        #cv2.rectangle(imageWhole, (boxesPID[k][0], boxesPID[k][1]), (boxesPID[k][0] + boxesPID[k][2], boxesPID[k][1] + boxesPID[k][3]), colorB, 7)
        #cv2.rectangle(imageOut, (boxesPID[k][0], boxesPID[k][1]), (boxesPID[k][0] + boxesPID[k][2], boxesPID[k][1] + boxesPID[k][3]), colorB, 7)
        
        
        #text = "{}: {:.4f}".format(LABELS[boxesPID[k][4]], boxesPID[k][5]) #labels with confidence values
        text = "{}".format(LABELS[boxesPID[k][4]]) #labels without confidence values
        #cv2.putText(imageWhole, text, (boxesPID[k][0], boxesPID[k][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colorB, 2)
   
       # cv2.putText(imageOut, text, (boxesPID[k][0], boxesPID[k][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colorB, 2)
     
  
    annotationFilename = pidNo + "_Annotations_v4.json"
    
    
    ##check if file is present 
    global filePresent
    filePresent = 0
    print(annotationFilename)
    if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], annotationFilename)):
        filePresent = 1
        print('file found for comparison of symbols detection', filePresent)
    else: 
        filePresent = 0
        print('no file found for comparison of symbols detection', filePresent)            
        
        
    
    if (filePresent==1):#store details from json file 
    
        with open(os.path.join(app.config['UPLOAD_FOLDER'], annotationFilename)) as f:
            data = json.load(f)
        
        

        
        #pidName = annotationFilename[:-20]#changed filename
        #nameForTextFile = pidName + ".txt"
        
        
        #timeValue = str(int(time.time()))      
        #global nameForCSVFile
        #nameForCSVFile = pidNo + "_" + timeValue + ".csv"   
        
        #fileCount = fileCount+1
        drawingNo = data[0]["filename"]
        listSamples = data[0]["annotations"]
        print(annotationFilename, "for P&ID", drawingNo, "containing", len(listSamples), "samples")
        #print(len(listSamples))  
        
        
        for n in range (0,len(listSamples)):
        
            xTemp = data[0]["annotations"][n]["x"]
            yTemp = data[0]["annotations"][n]["y"]
            widthTemp = data[0]["annotations"][n]["width"]
            heightTemp = data[0]["annotations"][n]["height"]
            classTemp = data[0]["annotations"][n]["class"]
            
            
            annotationsStoreList.append([drawingNo, xTemp, yTemp, widthTemp, heightTemp, classTemp])
    
    else:
        annotationsStoreList = []
        
    timeValue = str(int(time.time()))    
    
    global nameForCSVFileSymbols
    global nameForCSVFileSymbols_c
    
    global nameForCSVFile
    with open(os.path.join(app.config['UPLOAD_FOLDER'], nameForCSVFile), 'a', newline = '') as outfile, open(os.path.join(app.config['UPLOAD_FOLDER'], nameForCSVFileSymbols), 'a', newline = '') as outfileSymbol, open(os.path.join(app.config['UPLOAD_FOLDER'], nameForCSVFileSymbols_c), 'a', newline = '') as outfileSymbol_c:    
        csvWriter = csv.writer(outfile, delimiter = ',')
        csvWriterSymbols = csv.writer(outfileSymbol, delimiter = ',')
        csvWriterSymbols_c = csv.writer(outfileSymbol_c, delimiter = ',')
        
        print("filePresent", filePresent, "isCropped", isCropped)
        
        if (filePresent==1):  #if jsonFile
            
            #script = os.path.basename(__file__)for windows
            #csvWriter.writerow([script])

            csvWriter.writerow(['diagramNo', 'Labelled Class', 'x', 'y', 'width', 'height', 'Predicted Class', 'itemNumber', 'x', 'y', 'width', 'height', 'IOU', 'matching class'])
            csvWriterSymbols.writerow(['diagramNo', 'Labelled Class', 'x', 'y', 'width', 'height', 'Predicted Class', 'itemNumber', 'x', 'y', 'width', 'height', 'IOU', 'matching class'])
            csvWriterSymbols_c.writerow(['diagramNo', 'Labelled Class', 'x', 'y', 'width', 'height', 'Predicted Class', 'itemNumber', 'x', 'y', 'width', 'height', 'IOU', 'matching class', 'confidence'])

            listPredictionMatch = []
            listPredictionMatch_c = []
            
            for n in range(0,len(annotationsStoreList)): ##loop through the labelled boxes
                
                missed = 1
                listRow = []
                listRow = [annotationsStoreList[n][0], annotationsStoreList[n][5], annotationsStoreList[n][1], annotationsStoreList[n][2], annotationsStoreList[n][3], annotationsStoreList[n][4]]
                listRow_c = []
                listRow_c = [annotationsStoreList[n][0], annotationsStoreList[n][5], annotationsStoreList[n][1], annotationsStoreList[n][2], annotationsStoreList[n][3], annotationsStoreList[n][4]]
                                     
                #csvWriter.writerow(listRow)  
                #cv2.rectangle(imageWhole, (int(annotationsStoreList[n][1]), int(annotationsStoreList[n][2])), (int(annotationsStoreList[n][1] + annotationsStoreList[n][3]), int(annotationsStoreList[n][2] + annotationsStoreList[n][4])), (255,255,0), 3)
        
        #text = "{}: {:.4f}".format(LABELS[boxesPID[k][4]], boxesPID[k][5]) #labels with confidence values
        #text = "{}".format(LABELS[boxesPID[k][4]]) #labels without confidence values
        #cv2.putText(imageWhole, text, (boxesPID[k][0], boxesPID[k][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colorB, 2)
                
                box1 = [annotationsStoreList[n][1],annotationsStoreList[n][2],annotationsStoreList[n][3],annotationsStoreList[n][4]]#x,y,width,height box 1 #item in labelled data                      
                tempActualCentreX = annotationsStoreList[n][1] + ((annotationsStoreList[n][3])/2)
                tempActualCentreY = annotationsStoreList[n][2] + ((annotationsStoreList[n][4])/2)
                
                for k in range(len(boxesPID)): #loop through found detections                      
                    
                    box2 = [boxesPID[k][0], boxesPID[k][1], boxesPID[k][2], boxesPID[k][3]]
    
                    box1x2 = box1[0] + box1[2]
                    box1y2 = box1[1]+ box1[3]
                    
                    box2x2 = box2[0] + box2[2]
                    box2y2 = box2[1]+ box2[3]
                    
                    xLow = max(box1[0], box2[0]) ##max of box1 low or box 2 low
                    xHigh = min(box1x2, box2x2)
                    yLow = max(box1[1], box2[1])
                    yHigh = min(box1y2, box2y2) 
                    
                    iou = 0
                    
                    tempPredCentreX = boxesPID[k][0] + ((boxesPID[k][2])/2)
                    tempPredCentreY = boxesPID[k][1] + ((boxesPID[k][3])/2)
                    
                    if (xLow<=xHigh) or (yLow<=yHigh):
                    
                        areaOverlap = (xHigh-xLow) * (yHigh-yLow)
                        
                        areaBox1 = (box1x2 - box1[0])*(box1y2 - box1[1])
                        areaBox2 = (box2x2 - box2[0])*(box2y2 - box2[1])
                        
                        if (areaOverlap>0) and (areaOverlap<=(min(areaBox1, areaBox2))) :
                        
                            iou = areaOverlap/(areaBox1 + areaBox2 - areaOverlap)
                     
                    if (iou>=iou_threshold):
                        classTemp = LABELS[boxesPID[k][4]]
                        
                        listRow.append(classTemp)
                        listRow.append(boxesPID[k][6])##add the item Number
                        listRow.append(boxesPID[k][0])
                        listRow.append(boxesPID[k][1])
                        listRow.append(boxesPID[k][2])
                        listRow.append(boxesPID[k][3])
                        listRow.append(iou)     


                        listRow_c.append(classTemp)
                        listRow_c.append(boxesPID[k][6])##add the item Number
                        listRow_c.append(boxesPID[k][0])
                        listRow_c.append(boxesPID[k][1])
                        listRow_c.append(boxesPID[k][2])
                        listRow_c.append(boxesPID[k][3])
                        listRow_c.append(iou)                    
                        




                        listPredictionMatch.append([LABELS[boxesPID[k][4]], boxesPID[k][6], boxesPID[k][0], boxesPID[k][1], boxesPID[k][2], boxesPID[k][3]])
                        listPredictionMatch_c.append([LABELS[boxesPID[k][4]], boxesPID[k][6], boxesPID[k][0], boxesPID[k][1], boxesPID[k][2], boxesPID[k][3], boxesPID[k][5]])
                        
                        #if(classTemp == annotationsStoreList[n][5]):
                        if(' '.join(classTemp.split()) == ' '.join( (annotationsStoreList[n][5]).split() ) ) :

                            sameLabel = '1' #same class
                            listRow.append(sameLabel)
                            listRow_c.append(sameLabel)
                            #print('same', classTemp, annotationsStoreList[n][5])
                        else:
                            sameLabel = '0' #different class
                            listRow.append(sameLabel)
                            listRow_c.append(sameLabel)
                            #print('different', classTemp, annotationsStoreList[n][5])   

                        listRow_c.append(boxesPID[k][5])   ### append the confidence                          
                            
                        
                        missed = 0                       
                    
                if missed == 1:
                        listRow.append('0')
                        listRow.append('0')#extra for item number                        
                        listRow.append('0')
                        listRow.append('0')
                        listRow.append('0')
                        listRow.append('0')
                        listRow.append('0')
                        listRow.append('-1')           

                        listRow_c.append('0')
                        listRow_c.append('0')#extra for item number                        
                        listRow_c.append('0')
                        listRow_c.append('0')
                        listRow_c.append('0')
                        listRow_c.append('0')
                        listRow_c.append('0')
                        listRow_c.append('-1')                
                                              
                            
                csvWriter.writerow(listRow)                            
                csvWriterSymbols.writerow(listRow)
                csvWriterSymbols_c.writerow(listRow_c)             
                    #csvWriter.writerow(listRow)     
                
            noActualSymbols = len(annotationsStoreList)
            noPredictedSymbols = len(boxesPID)
            
            csvWriter.writerow(['no. of actual symbols:', noActualSymbols, 'no. of predicted symbols:', noPredictedSymbols])
            csvWriter.writerow(['list of extra detections not matching with ground truth:'])
            csvWriter.writerow(['predicted symbol', 'item number',  'x', 'y', 'width', 'height'])

            csvWriterSymbols.writerow(['no. of actual symbols:', noActualSymbols, 'no. of predicted symbols:', noPredictedSymbols])
            csvWriterSymbols.writerow(['list of extra detections not matching with ground truth:'])
            csvWriterSymbols.writerow(['predicted symbol', 'item number',  'x', 'y', 'width', 'height'])

            csvWriterSymbols_c.writerow(['no. of actual symbols:', noActualSymbols, 'no. of predicted symbols:', noPredictedSymbols])
            csvWriterSymbols_c.writerow(['list of extra detections not matching with ground truth:'])
            csvWriterSymbols_c.writerow(['predicted symbol', 'item number',  'x', 'y', 'width', 'height'])
            
            
    
            listPred = []
            listPred_c = []
            import numpy
            a = numpy.asarray(boxesPID)
            #numpy.savetxt("foo.csv", a, delimiter=",")
            for k in range(len(boxesPID)):        
                    classTemp = LABELS[boxesPID[k][4]]                
                    listPred.append([classTemp, boxesPID[k][6], boxesPID[k][0], boxesPID[k][1], boxesPID[k][2], boxesPID[k][3]])
                    listPred_c.append([classTemp, boxesPID[k][6], boxesPID[k][0], boxesPID[k][1], boxesPID[k][2], boxesPID[k][3], boxesPID[k][5]])

            differences=[]
            
            for list in listPred:
                if list not in listPredictionMatch:
                    differences.append(list)
                    #print('found false postive that is not matching', list)

            differences_c=[]
            
            for list in listPred_c:
                if list not in listPredictionMatch_c:
                    differences_c.append(list)
                    #print('found false postive that is not matching', list)
            
            for n in range (0, len(differences)):
                extraDetections = []
                extraDetections=[differences[n][0], differences[n][1], differences[n][2], differences[n][3], differences[n][4], differences[n][5]]# extra added in for item number 
                csvWriter.writerow(extraDetections)
                csvWriterSymbols.writerow(extraDetections)     


            for n in range (0, len(differences_c)):   
                extraDetections_c = []
                extraDetections_c=[differences_c[n][0], differences_c[n][1], differences_c[n][2], differences_c[n][3], differences_c[n][4], differences_c[n][5], differences_c[n][6]]        
                csvWriterSymbols_c.writerow(extraDetections_c)          




        else:

            
            csvWriter.writerow(['Predicted Class', 'item number', 'x', 'y', 'width', 'height'])
            csvWriterSymbols.writerow(['Predicted Class', 'item number', 'x', 'y', 'width', 'height'])  
            csvWriterSymbols_c.writerow(['Predicted Class', 'item number', 'x', 'y', 'width', 'height', 'confidence'])     
            
            for k in range(len(boxesPID)): 
                classTemp = LABELS[boxesPID[k][4]]                
                csvWriter.writerow([classTemp, boxesPID[k][6], boxesPID[k][0], boxesPID[k][1], boxesPID[k][2], boxesPID[k][3]])
                csvWriterSymbols.writerow([classTemp, boxesPID[k][6], boxesPID[k][0], boxesPID[k][1], boxesPID[k][2], boxesPID[k][3]])
                csvWriterSymbols_c.writerow([classTemp, boxesPID[k][6], boxesPID[k][0], boxesPID[k][1], boxesPID[k][2], boxesPID[k][3], boxesPID[k][5]])


    
    filenamePID = pidNo + "_detection.png"
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], filenamePID), imageOut) ##**save the patch with the detections here     
    
    #filenamePIDGrid = pidNo + "_detection_grid.png"
    #cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], filenamePIDGrid), imageWholeGrid) ##**save the patch with the detections here 
    
    timeValue = str(int(time.time()))      
    global filenamePIDTimeS
    filenamePIDTimeS = pidNo + "_" + timeValue + "_detection.png"
    
    #print("filenamePIDTimeS is : ", filenamePIDTimeS)
    print("item no after detect symbols:", itemNo)

    global totalSymbols
    totalSymbols = itemNo
    global totalSymbolsText
    totalSymbolsText = itemNo

    
    #global avSD
    #avSD = sum(sensorD)/len(sensorD)
    
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], filenamePIDTimeS), imageOut) ##**save the patch with the detections here     

    global boxesPID_updated
    global symbolClassList

    symbolClassList = []

    # for p in range(len(classIDs)):
    #     symbolClassList.append(LABELS[classIDs[p]])

    for q in range(len(LABELS)):
        symbolClassList.append(LABELS[q])




 
    for k in range(len(boxesPID)):
        boxesPID_updated.append([boxesPID[k]])

    print('in detectSymbols, length of boxesPID:, length of boxesPID_updated: ', len(boxesPID), len(boxesPID_updated))

## detectSymbols__ function ###################################################################################################################################
## function to detect symbols in drawing ################################################################################################################################### 
def detectSymbols__():
    
    print("diagram name is: ", currentPID)
    
    filename = currentPID
    filenameToReadOutputS = currentPID[:-4] + "_detection.png"
    imageOut = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameToReadOutputS))    
    
    filenameToRead = currentPID  ##reads the uploaded file   
    imageWhole = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameToRead))
    imageWholeGrid = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameToRead))

    height, width = imageWhole.shape[:2]
    
    imageA1 = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameToRead)) 
    imageA2 = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameToRead)) 
        
    dH = 4
    windowsize_r = int(height/4) - 1 #height -1 for rounding error to allow grid of 4 x 6
    
    if (windowsize_r>1700):
        windowsize_r = int(height/5) - 1 
        dH = 5
        if (windowsize_r>1700):
            windowsize_r = int(height/6) - 1 
            dH = 6
            if (windowsize_r>1700):
                windowsize_r = int(height/7) - 1 
                dH = 7
    
    if (windowsize_r<780):
        windowsize_r = int(height/3) - 1 
        dH = 3
        if (windowsize_r<780):
            windowsize_r = int(height/2) - 1 
            dH = 2
            if (windowsize_r<780):
                windowsize_r = height - 1
                dH = 1
                            
    dW = 6                        
    windowsize_c = int(width/6) - 1 #width -1 for rounding error to allow grid of 4 x 6
    
    if (windowsize_c>1700):
        windowsize_c = int(width/7) - 1 
        dW = 7
        if (windowsize_c>1700):
            windowsize_c = int(width/8) - 1 
            dW = 8
            if (windowsize_c>1700):
                windowsize_c = int(width/9) - 1 
                dW = 9
    
    if (windowsize_c<750):
            windowsize_c = int(width/5) - 1 
            dW = 5
            if (windowsize_c<750):
                windowsize_c = int(width/4) - 1 
                dW = 4
                if (windowsize_c<750):
                    windowsize_c = int(width/3) - 1 
                    dW = 3
                    if (windowsize_c<750):
                        windowsize_c = int(width/2) - 1
                        dW = 2
                        if (windowsize_c<750):
                            windowsize_c = width - 1
                            dW = 1

    

    pidNo = filename[:-4]
    global boxesPID
    boxesPID = [] #list for boxes in the full image storing x,y,w,h,classID,confidence

    
    image_patch = imageWhole

    #pathToYoloFolder = "yolo-symbol" ############
    pathToYoloFolder = "yolo-symbolv2" ############
    confidenceMin = conf_threshold
    thresholdNMS = 0.3

    #labelsPath = os.path.sep.join([pathToYoloFolder, "24Symbols.names"])############
    labelsPath = os.path.sep.join([pathToYoloFolder, names_file])############
    global LABELS
    LABELS = open(labelsPath).read().strip().split("\n")
    
    np.random.seed(42)
    global COLORS
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
        dtype="uint8")
    #print (*COLORS, sep = '\n')

    weightsPath = os.path.sep.join([pathToYoloFolder, weights_file])############
    configPath = os.path.sep.join([pathToYoloFolder, cfg_file])############
    
    print("starting object detection")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
   
    image = image_patch
    (H, W) = image.shape[:2]
    
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (2400, 2400),  
        swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
     
    
    boxes = []
    confidences = []
    classIDs = []
    
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
     
            if confidence > confidenceMin:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
     
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
     
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidenceMin, thresholdNMS)
    
    global itemNo
    global scanAreaY
    global scanAreaX
    global highestXS
    global lowestXS
    global highestYX
    global lowestYX
    global symbolNumber

    itemNo = 0    
    lowestXS = W
    lowestYS = H
    highestXS = 0
    highestYS = 0
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            #xFullImage = c + x
            #yFullImage = r + y
            xFullImage = x
            yFullImage = y
            (wFullImage, hFullImage) = (w, h)
            
            itemNo = itemNo + 1
            itemNumber = itemNo          
            boxesPID.append([xFullImage, yFullImage, wFullImage, hFullImage, classIDs[i], confidences[i], itemNumber])         
            

            #color = [int(c) for c in COLORS[classIDs[i]]]
            #cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(imageA1, (x, y), (x + w, y + h), (0, 0, 255), 4)   
            cv2.rectangle(imageA1, (x, y), (x + w, y + h), (255, 255, 255), -1)          
            cv2.rectangle(imageA2, (x, y), (x + w, y + h), (255, 255, 255), -1)
            
            if((x + w)> highestXS):            
                highestXS = x + w   
                symbol = LABELS[classIDs[i]]

            if((x)< lowestXS):            
                lowestXS = x   

            if((y + h)> highestYS):            
                highestYS = y + h   
                
            if((y)< lowestYS):            
                lowestYS = y  

            # #if(classIDs[i] == 27) or (classIDs[i] == 28) or (classIDs[i] == 0): 
            # if(classIDs[i] != 7) or (classIDs[i] != 26): 
            #     #stempCentre = y + int(h/2)
            #     tempYL = y + int(h/4)
            #     tempYH = y + int(3*(h/4))
            #     #scanAreaY.append([y,y+h])                 
            #     scanAreaY.append([tempYL,tempYH])        
                
                
            # if(classIDs[i] != 26) or (classIDs[i] != 27) or (classIDs[i] != 28): 
            #     #stempCentre = y + int(h/2)
            #     tempYL = y + int(h/4)
            #     tempYH = y + int(3*(h/4))
            #     #scanAreaY.append([y,y+h])                 
            #     scanAreaX.append([tempYL,tempYH])         
                
            #text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            #cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
     
    #filenameDetect = filename[:-4] + "_result.png"   ##******change the filename here
    #filenameDetect = patchFilename[:-4] + "_result.png"   ##******change the filename here
    #cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], filenameDetect), image) ##**save the patch with the detections here 
    
    print("x limits:", lowestXS, highestXS, "y limits:", lowestYS, highestYS)  
  
    imTest = image
    
    cv2.rectangle(imTest, (lowestXS, lowestYS), (highestXS, highestYS), (255, 255, 255), -1)
    
    imTest2 = image
    cv2.rectangle(imTest2, (lowestXS, lowestYS), (highestXS, highestYS), (255, 0, 255), 3)

    # filenameimTest = "imTest.png"
    # cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], filenameimTest), imTest)


    # filenameimTest2 = "imTest2.png"
    # cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], filenameimTest2), imTest2)
    
    
    
    #print("highestXS is:", highestXS, "symbol is: ", symbol)
    
    filenameImageA1 = currentPID[:-4] + "_A1.png"   ##******change the filename here
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], filenameImageA1), imageA1) ##**save the patch with the detections here     
  
    filenameImageA2 = currentPID[:-4] + "_A2.png"   ##******change the filename here
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], filenameImageA2), imageA2) ##**save the patch with the detections here 

    symbolNumber = itemNo
    
#####opencv end
    global annotationsStoreList
    #global sensorD
    #print('boxesPID length', len(boxesPID))
    #print('boxesPID', boxesPID)
    #print("itemNo after symbols csv print:", itemNo)    
    for k in range(len(boxesPID)):
        
        colorB = [int(c) for c in COLORS[boxesPID[k][4]]]
        
        # if (LABELS[boxesPID[k][4]] == "Sensor"):
        #     colorB = (0, 0, 255)
        #     sensorD.append(boxesPID[k][2])
            
        #cv2.rectangle(imageWhole, (boxesPID[k][0], boxesPID[k][1]), (boxesPID[k][0] + boxesPID[k][2], boxesPID[k][1] + boxesPID[k][3]), colorB, 7)
        #cv2.rectangle(imageOut, (boxesPID[k][0], boxesPID[k][1]), (boxesPID[k][0] + boxesPID[k][2], boxesPID[k][1] + boxesPID[k][3]), colorB, 7)
        
        
        #text = "{}: {:.4f}".format(LABELS[boxesPID[k][4]], boxesPID[k][5]) #labels with confidence values
        text = "{}".format(LABELS[boxesPID[k][4]]) #labels without confidence values
        #cv2.putText(imageWhole, text, (boxesPID[k][0], boxesPID[k][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colorB, 2)
   
       # cv2.putText(imageOut, text, (boxesPID[k][0], boxesPID[k][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colorB, 2)
     
  
    annotationFilename = pidNo + "_Annotations_v4.json"
    
    
    ##check if file is present 
    global filePresent
    filePresent = 0
    
    if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], annotationFilename)):
        filePresent = 1
        print('file found for comparison of symbols detection', filePresent)
    else: 
        filePresent = 0
        print('no file found for comparison of symbols detection', filePresent)            
        
        
    
    if (filePresent==1):#store details from json file 
    
        with open(os.path.join(app.config['UPLOAD_FOLDER'], annotationFilename)) as f:
            data = json.load(f)
        
        

        
        #pidName = annotationFilename[:-20]#changed filename
        #nameForTextFile = pidName + ".txt"
        
        
        #timeValue = str(int(time.time()))      
        #global nameForCSVFile
        #nameForCSVFile = pidNo + "_" + timeValue + ".csv"   
        
        #fileCount = fileCount+1
        drawingNo = data[0]["filename"]
        listSamples = data[0]["annotations"]
        print(annotationFilename, "for P&ID", drawingNo, "containing", len(listSamples), "samples")
        #print(len(listSamples))  
        
        
        for n in range (0,len(listSamples)):
        
            xTemp = data[0]["annotations"][n]["x"]
            yTemp = data[0]["annotations"][n]["y"]
            widthTemp = data[0]["annotations"][n]["width"]
            heightTemp = data[0]["annotations"][n]["height"]
            classTemp = data[0]["annotations"][n]["class"]
            
            
            annotationsStoreList.append([drawingNo, xTemp, yTemp, widthTemp, heightTemp, classTemp])
    
    else:
        annotationsStoreList = []
        
    timeValue = str(int(time.time()))    
    
    global nameForCSVFileSymbols
    global nameForCSVFileSymbols_c
    
    global nameForCSVFile
    with open(os.path.join(app.config['UPLOAD_FOLDER'], nameForCSVFile), 'a', newline = '') as outfile, open(os.path.join(app.config['UPLOAD_FOLDER'], nameForCSVFileSymbols), 'a', newline = '') as outfileSymbol, open(os.path.join(app.config['UPLOAD_FOLDER'], nameForCSVFileSymbols_c), 'a', newline = '') as outfileSymbol_c:    
        csvWriter = csv.writer(outfile, delimiter = ',')
        csvWriterSymbols = csv.writer(outfileSymbol, delimiter = ',')
        csvWriterSymbols_c = csv.writer(outfileSymbol_c, delimiter = ',')
        
        print("filePresent", filePresent, "isCropped", isCropped)
        
        if (filePresent==1):  #if jsonFile
            
            #script = os.path.basename(__file__)for windows
            #csvWriter.writerow([script])

            csvWriter.writerow(['diagramNo', 'Labelled Class', 'x', 'y', 'width', 'height', 'Predicted Class', 'itemNumber', 'x', 'y', 'width', 'height', 'IOU', 'matching class'])
            csvWriterSymbols.writerow(['diagramNo', 'Labelled Class', 'x', 'y', 'width', 'height', 'Predicted Class', 'itemNumber', 'x', 'y', 'width', 'height', 'IOU', 'matching class'])
            csvWriterSymbols_c.writerow(['diagramNo', 'Labelled Class', 'x', 'y', 'width', 'height', 'Predicted Class', 'itemNumber', 'x', 'y', 'width', 'height', 'IOU', 'matching class', 'confidence'])

            listPredictionMatch = []
            listPredictionMatch_c = []
            
            for n in range(0,len(annotationsStoreList)): ##loop through the labelled boxes
                
                missed = 1
                listRow = []
                listRow = [annotationsStoreList[n][0], annotationsStoreList[n][5], annotationsStoreList[n][1], annotationsStoreList[n][2], annotationsStoreList[n][3], annotationsStoreList[n][4]]
                listRow_c = []
                listRow_c = [annotationsStoreList[n][0], annotationsStoreList[n][5], annotationsStoreList[n][1], annotationsStoreList[n][2], annotationsStoreList[n][3], annotationsStoreList[n][4]]
                                     
                #csvWriter.writerow(listRow)  
                #cv2.rectangle(imageWhole, (int(annotationsStoreList[n][1]), int(annotationsStoreList[n][2])), (int(annotationsStoreList[n][1] + annotationsStoreList[n][3]), int(annotationsStoreList[n][2] + annotationsStoreList[n][4])), (255,255,0), 3)
        
        #text = "{}: {:.4f}".format(LABELS[boxesPID[k][4]], boxesPID[k][5]) #labels with confidence values
        #text = "{}".format(LABELS[boxesPID[k][4]]) #labels without confidence values
        #cv2.putText(imageWhole, text, (boxesPID[k][0], boxesPID[k][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colorB, 2)
                
                box1 = [annotationsStoreList[n][1],annotationsStoreList[n][2],annotationsStoreList[n][3],annotationsStoreList[n][4]]#x,y,width,height box 1 #item in labelled data                      
                tempActualCentreX = annotationsStoreList[n][1] + ((annotationsStoreList[n][3])/2)
                tempActualCentreY = annotationsStoreList[n][2] + ((annotationsStoreList[n][4])/2)
                
                for k in range(len(boxesPID)): #loop through found detections                      
                    
                    box2 = [boxesPID[k][0], boxesPID[k][1], boxesPID[k][2], boxesPID[k][3]]
    
                    box1x2 = box1[0] + box1[2]
                    box1y2 = box1[1]+ box1[3]
                    
                    box2x2 = box2[0] + box2[2]
                    box2y2 = box2[1]+ box2[3]
                    
                    xLow = max(box1[0], box2[0]) ##max of box1 low or box 2 low
                    xHigh = min(box1x2, box2x2)
                    yLow = max(box1[1], box2[1])
                    yHigh = min(box1y2, box2y2) 
                    
                    iou = 0
                    
                    tempPredCentreX = boxesPID[k][0] + ((boxesPID[k][2])/2)
                    tempPredCentreY = boxesPID[k][1] + ((boxesPID[k][3])/2)
                    
                    if (xLow<=xHigh) or (yLow<=yHigh):
                    
                        areaOverlap = (xHigh-xLow) * (yHigh-yLow)
                        
                        areaBox1 = (box1x2 - box1[0])*(box1y2 - box1[1])
                        areaBox2 = (box2x2 - box2[0])*(box2y2 - box2[1])
                        
                        if (areaOverlap>0) and (areaOverlap<=(min(areaBox1, areaBox2))) :
                        
                            iou = areaOverlap/(areaBox1 + areaBox2 - areaOverlap)
                     
                    if (iou>=iou_threshold):
                        classTemp = LABELS[boxesPID[k][4]]
                        
                        listRow.append(classTemp)
                        listRow.append(boxesPID[k][6])##add the item Number
                        listRow.append(boxesPID[k][0])
                        listRow.append(boxesPID[k][1])
                        listRow.append(boxesPID[k][2])
                        listRow.append(boxesPID[k][3])
                        listRow.append(iou)     


                        listRow_c.append(classTemp)
                        listRow_c.append(boxesPID[k][6])##add the item Number
                        listRow_c.append(boxesPID[k][0])
                        listRow_c.append(boxesPID[k][1])
                        listRow_c.append(boxesPID[k][2])
                        listRow_c.append(boxesPID[k][3])
                        listRow_c.append(iou)                    
                        




                        listPredictionMatch.append([LABELS[boxesPID[k][4]], boxesPID[k][6], boxesPID[k][0], boxesPID[k][1], boxesPID[k][2], boxesPID[k][3]])
                        listPredictionMatch_c.append([LABELS[boxesPID[k][4]], boxesPID[k][6], boxesPID[k][0], boxesPID[k][1], boxesPID[k][2], boxesPID[k][3], boxesPID[k][5]])
                        
                        #if(classTemp == annotationsStoreList[n][5]):
                        if(' '.join(classTemp.split()) == ' '.join( (annotationsStoreList[n][5]).split() ) ) :

                            sameLabel = '1' #same class
                            listRow.append(sameLabel)
                            listRow_c.append(sameLabel)
                            #print('same', classTemp, annotationsStoreList[n][5])
                        else:
                            sameLabel = '0' #different class
                            listRow.append(sameLabel)
                            listRow_c.append(sameLabel)
                            #print('different', classTemp, annotationsStoreList[n][5])   

                        listRow_c.append(boxesPID[k][5])   ### append the confidence                          
                            
                        
                        missed = 0                       
                    
                if missed == 1:
                        listRow.append('0')
                        listRow.append('0')#extra for item number                        
                        listRow.append('0')
                        listRow.append('0')
                        listRow.append('0')
                        listRow.append('0')
                        listRow.append('0')
                        listRow.append('-1')           

                        listRow_c.append('0')
                        listRow_c.append('0')#extra for item number                        
                        listRow_c.append('0')
                        listRow_c.append('0')
                        listRow_c.append('0')
                        listRow_c.append('0')
                        listRow_c.append('0')
                        listRow_c.append('-1')                
                                              
                            
                csvWriter.writerow(listRow)                            
                csvWriterSymbols.writerow(listRow)
                csvWriterSymbols_c.writerow(listRow_c)             
                    #csvWriter.writerow(listRow)     
                
            noActualSymbols = len(annotationsStoreList)
            noPredictedSymbols = len(boxesPID)
            
            csvWriter.writerow(['no. of actual symbols:', noActualSymbols, 'no. of predicted symbols:', noPredictedSymbols])
            csvWriter.writerow(['list of extra detections not matching with ground truth:'])
            csvWriter.writerow(['predicted symbol', 'item number',  'x', 'y', 'width', 'height'])

            csvWriterSymbols.writerow(['no. of actual symbols:', noActualSymbols, 'no. of predicted symbols:', noPredictedSymbols])
            csvWriterSymbols.writerow(['list of extra detections not matching with ground truth:'])
            csvWriterSymbols.writerow(['predicted symbol', 'item number',  'x', 'y', 'width', 'height'])

            csvWriterSymbols_c.writerow(['no. of actual symbols:', noActualSymbols, 'no. of predicted symbols:', noPredictedSymbols])
            csvWriterSymbols_c.writerow(['list of extra detections not matching with ground truth:'])
            csvWriterSymbols_c.writerow(['predicted symbol', 'item number',  'x', 'y', 'width', 'height'])
            
            
    
            listPred = []
            listPred_c = []
            import numpy
            a = numpy.asarray(boxesPID)
            #numpy.savetxt("foo.csv", a, delimiter=",")
            for k in range(len(boxesPID)):        
                    classTemp = LABELS[boxesPID[k][4]]                
                    listPred.append([classTemp, boxesPID[k][6], boxesPID[k][0], boxesPID[k][1], boxesPID[k][2], boxesPID[k][3]])
                    listPred_c.append([classTemp, boxesPID[k][6], boxesPID[k][0], boxesPID[k][1], boxesPID[k][2], boxesPID[k][3], boxesPID[k][5]])

            differences=[]
            
            for list in listPred:
                if list not in listPredictionMatch:
                    differences.append(list)
                    #print('found false postive that is not matching', list)

            differences_c=[]
            
            for list in listPred_c:
                if list not in listPredictionMatch_c:
                    differences_c.append(list)
                    #print('found false postive that is not matching', list)
            
            for n in range (0, len(differences)):
                extraDetections = []
                extraDetections=[differences[n][0], differences[n][1], differences[n][2], differences[n][3], differences[n][4], differences[n][5]]# extra added in for item number 
                csvWriter.writerow(extraDetections)
                csvWriterSymbols.writerow(extraDetections)     


            for n in range (0, len(differences_c)):   
                extraDetections_c = []
                extraDetections_c=[differences_c[n][0], differences_c[n][1], differences_c[n][2], differences_c[n][3], differences_c[n][4], differences_c[n][5], differences_c[n][6]]        
                csvWriterSymbols_c.writerow(extraDetections_c)          




        else:

            
            csvWriter.writerow(['Predicted Class', 'item number', 'x', 'y', 'width', 'height'])
            csvWriterSymbols.writerow(['Predicted Class', 'item number', 'x', 'y', 'width', 'height'])  
            csvWriterSymbols_c.writerow(['Predicted Class', 'item number', 'x', 'y', 'width', 'height', 'confidence'])     
            
            for k in range(len(boxesPID)): 
                classTemp = LABELS[boxesPID[k][4]]                
                csvWriter.writerow([classTemp, boxesPID[k][6], boxesPID[k][0], boxesPID[k][1], boxesPID[k][2], boxesPID[k][3]])
                csvWriterSymbols.writerow([classTemp, boxesPID[k][6], boxesPID[k][0], boxesPID[k][1], boxesPID[k][2], boxesPID[k][3]])
                csvWriterSymbols_c.writerow([classTemp, boxesPID[k][6], boxesPID[k][0], boxesPID[k][1], boxesPID[k][2], boxesPID[k][3], boxesPID[k][5]])
    
    filenamePID = pidNo + "_detection.png"
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], filenamePID), imageOut) ##**save the patch with the detections here     
    
    #filenamePIDGrid = pidNo + "_detection_grid.png"
    #cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], filenamePIDGrid), imageWholeGrid) ##**save the patch with the detections here 
    
    timeValue = str(int(time.time()))      
    global filenamePIDTimeS
    filenamePIDTimeS = pidNo + "_" + timeValue + "_detection.png"
    
    #print("filenamePIDTimeS is : ", filenamePIDTimeS)
    print("item no after detect symbols:", itemNo)

    global totalSymbols
    totalSymbols = itemNo
    global totalSymbolsText
    totalSymbolsText = itemNo

    
    #global avSD
    #avSD = sum(sensorD)/len(sensorD)
    
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], filenamePIDTimeS), imageOut) ##**save the patch with the detections here     

    global boxesPID_updated
    global symbolClassList

    symbolClassList = []

    # for p in range(len(classIDs)):
    #     symbolClassList.append(LABELS[classIDs[p]])

    for q in range(len(LABELS)):
        symbolClassList.append(LABELS[q])




 
    for k in range(len(boxesPID)):
        boxesPID_updated.append([boxesPID[k]])

    print('in detectSymbols, length of boxesPID:, length of boxesPID_updated: ', len(boxesPID), len(boxesPID_updated))

## createChangeCSV function ###################################################################################################################################
## function to create csv ################################################################################################################################### 
def createChangeCSV():

    #print('number of predicted symbols removed: ', len(removedSName))
    #print('number of symbols added: ', len(addedSName))

    global approvedSymbols
    global removedSymbols
    global changedSymbols
    global changedSymbolsName

    global nameForChangeCSV
    nameForChangeCSV = currentPID[:-4] + "_Symbol_changes.csv"
    with open(os.path.join(app.config['UPLOAD_FOLDER'], nameForChangeCSV), 'w', newline = '') as outfileC:
        csvWriterChanges = csv.writer(outfileC, delimiter = ',')
        csvWriterChanges.writerow(['Alterations to the predicted symbols for diagram: ', currentPID, 'height:', origH, 'width:', origW])

    # rewrite to the csv
    with open(os.path.join(app.config['UPLOAD_FOLDER'], nameForChangeCSV), 'a', newline = '') as outfileC:    
        csvWriterChanges = csv.writer(outfileC, delimiter = ',')
        csvWriterChanges.writerow(['Removed symbols'])
        csvWriterChanges.writerow(['drawingName', 'Flag', 'PredictedClass', 'itemNumber', 'x', 'y', 'width', 'height'])

        #boxesPID [xFullImage 0, yFullImage 1, wFullImage 2, hFullImage 3, classIDs[i] 4, confidences[i] 5, itemNumber 6]
        for t in range(0, len(removedSymbols)):
            tempName = LABELS [ removedSymbols[t][4] ]
            tempItemNo = removedSymbols[t][6]
            tempX = removedSymbols[t][0]
            tempY = removedSymbols[t][1]
            tempW = removedSymbols[t][2]
            tempH = removedSymbols[t][3]
        
            csvWriterChanges.writerow([currentPID, -1, tempName, tempItemNo, tempX, tempY, tempW, tempH])

        csvWriterChanges.writerow(['Approved symbols'])
        csvWriterChanges.writerow(['drawingName', 'Flag', 'PredictedClass', 'itemNumber', 'x', 'y', 'width', 'height'])

        #boxesPID [xFullImage, yFullImage, wFullImage, hFullImage, classIDs[i], confidences[i], itemNumber]
        for t in range(0, len(approvedSymbols)):
            tempName = LABELS [ approvedSymbols[t][4] ]
            tempItemNo = approvedSymbols[t][6]
            tempX = approvedSymbols[t][0]
            tempY = approvedSymbols[t][1]
            tempW = approvedSymbols[t][2]
            tempH = approvedSymbols[t][3]
        
            csvWriterChanges.writerow([currentPID, 1, tempName, tempItemNo, tempX, tempY, tempW, tempH])

        csvWriterChanges.writerow(['Corrected class symbols'])
        csvWriterChanges.writerow(['drawingName', 'Flag', 'PredictedClass', 'itemNumber', 'x', 'y', 'width', 'height', 'UserDefinedClass'])

        for t in range(0, len(changedSymbolsName)):
            predictedName = LABELS [ changedSymbolsName[t][4] ]
            tempItemNo = changedSymbolsName[t][6]
            tempX = changedSymbolsName[t][0]
            tempY = changedSymbolsName[t][1]
            tempW = changedSymbolsName[t][2]
            tempH = changedSymbolsName[t][3]
            tempUserName = changedSymbolsName[t][7]
        
            csvWriterChanges.writerow([currentPID, 2, predictedName, tempItemNo, tempX, tempY, tempW, tempH, tempUserName])


    return 

#
## index function ###################################################################################################################################
## flask function  ################################################################################################################################### 
@app.route('/')
#@nocache
def index():
    
    return render_template('pIndex_B.html')
    
## uploadR function ###################################################################################################################################
## function that will redirect to index page  ################################################################################################################################### 
@app.route('/uploadR', methods=['POST', 'GET'])
#@nocache
def uploadR():

    print("new diagram")
    
    global lines
    global pipelines
    global lines2
    global results
    global boxesPID 
    global boxesPID_updated
    global boxesFull
    global confidencesFull
    global classIDsFull
    global COLORS
    global LABELS
    global arrows
    global pipes
    global xC
    global yC
    global wC
    global hC  
    global pX
    global pY
    global itemList
    global symbolCount
    global itemNo #csvNumber
    global detectNo  ##displayNumber
    global pipesDetected
    global aa
    global addedSName
    global removedSName
    global qt
    global isCropped
    global drawingNumber
    global numberBox   
    global drawingTitle
    global titleBox 
    global xGrid
    global yGrid
    global symbolClassList
    global filePresent
    global annotationsStoreList
    global nameForChangeCSV
    global symbolList
    global symbolRemove
    global symbolClassSelected
    global symbolClassIndex
    global removedSymbols
    global addedSymbols
    global approvedSymbols
    global changedSymbols
    global changedSymbolsName
    global currentPID
    global imageShown
    global filenamePIDTimeS
    global nameForCSVFile
    global nameForCSVFileSymbols
    global nameForCSVFileSymbols_c
    global g,k,m 
    global a,b,c
    global d,e,f
    global C1, C2, C3
    global scanAreaY
    global scanAreaX
    global origH
    global origW
    global highestXS
    global lowestXS
    global highestYX
    global lowestYX
    global panX 
    global panY
    global sf
    global symbolNumber
    global symbol
    global t
    global pipeline
    global dashes
    global lineSizesH
    global lineSizesV
    global nameForCSVFileText
    global nameForCSVFilePipelines
    global nameForCSVFileTextOnly
    global totalSymbols
    global totalSymbolsText
    global textJoin
    global totalItems

    global high_pred
    global low_pred
    global manual_checked_symbols 

    high_pred = []
    low_pred = []
    manual_checked_symbols = []

    dashes=[]
    lineSizesH = []
    lineSizesV = []
    nameForCSVFileText = ''
    nameForCSVFilePipelines = ''
    nameForCSVFileTextOnly = ''
    totalSymbols = 0
    totalSymbolsText = 0
    textJoin =[]
    totalItems = 0


    symbol=1
    t=1
    pipeline=1

    panX = 0 
    panY = 0
    sf = 1

    currentPID = "blank"
    imageShown =  "image"
    filenamePIDTimeS = ''
    nameForCSVFile = ''
    nameForCSVFileSymbols = ''
    nameForCSVFileSymbols_c = ''
    g,k,m = 1,1,1
    a,b,c = 1,1,1
    d,e,f = 1,1,1
    C1,C2,C3 = 0,0,0
    scanAreaY = []
    scanAreaX = []
    origH = 7000
    origW = 7000
    highestXS = 0
    lowestXS = 0
    highestYX = 0
    lowestYX = 0
    symbolNumber = 0

    removedSymbols = []
    addedSymbols = []
    changedSymbols = []
    changedSymbolsName = []
    approvedSymbols = []
    symbolClassList = []
    symbolClassSelected = 0
    symbolClassIndex = 0

    symbolRemove = -1

    symbolList = []

    filePresent = 0
    annotationsStoreList = []
    nameForChangeCSV = ''
    
    xGrid = 0
    yGrid = 0
    
    isCropped = 0
    qt=0
    addedSName = []
    removedSName = []
    aa=1
    pipesDetected = []
    itemNo = 0 #csvNumber
    detectNo = 0  ##displayNumber

    symbolCount = 0
    itemList = []
    lines = []
    pipelines = []
    lines2 = []
    results = []
    boxesPID = [] 
    boxesPID_updated = []
    COLORS=[]
    LABELS=[]
    arrows=[]   
    pipes = []
    xC = 0
    yC = 0
    wC = 0
    hC = 0
    pX = 0
    pY = 0
    drawingNumber = ""
    numberBox = []
    
    drawingTitle = ""
    titleBox = []

    boxesFull = []
    confidencesFull = []
    classIDsFull = []
    
    return render_template('pIndex_B.html')




## nearestS function ###################################################################################################################################
## function to find the closest symbol ################################################################################################################################### 

def nearestS(n3,n4,n5,n6,action,text):
    
    global annotationsStoreList
    global boxesPID
    global addedSName
    global removedSName
    global changedSymbols
    global changedSymbolsName
    global removedSymbols
    global addedSymbols
    global symbolNumber
    global manual_checked_symbols
    
    print("no of symbols before changes: ", len(boxesPID),"specified action: ", action, ' name: ', text)
    
    x1 = int(n3)#box
    y1 = int(n4)
    x2 = int(n5)
    y2 = int(n6)
    action = int(action)

    print("box", x1,y1,x2,y2)
    
    symbolI0 = 0
    symbolR =0
    r=0

    ## if 

    lowPredNR = []
    lowPredNR = [x for x in low_pred if x not in removedSymbols] # remove the removed symbols
    lowPredNR = [x for x in lowPredNR if x not in changedSymbols] # remove the removed symbols
    lowPredNR = [x for x in lowPredNR if x not in manual_checked_symbols] # remove the removed symbols

    if (abs(x2-x1)>0) and (abs(y2-y1)>0):

        if (action==4):# approve

            sRemove = 'empty'
            symbolI0 = 0
            sList = []
            switch = 0
            
            for p in range(0, len(lowPredNR)):##find nearest symbol to remove, from predicted and added symbols

                #if (LABELS[boxesPID[p][4]] == symbolClassSelected): #match to a symbol from the chosen class   
                if (x1<=lowPredNR[p][0]<=x2) and (y1<=lowPredNR[p][1]<=y2) and (x1<=(lowPredNR[p][0] + lowPredNR[p][2])<=x2) and (y1<=(lowPredNR[p][1] + lowPredNR[p][3])<=y2):
                
                    symbolI0 = symbolI0 + 1
                    symbolR = p # index of the symbol in user defined box 
                    sList.append(p)
                    switch = 1


            if (symbolI0==1): # if only 1 symbol was in the user defined box:

                if (switch==1):

                    sRemove = LABELS[lowPredNR[symbolR][4]] 
                    #removedSName.append([ lowPredNR[symbolR][0], lowPredNR[symbolR][1], lowPredNR[symbolR][2], lowPredNR[symbolR][3], lowPredNR[symbolR][4], lowPredNR[symbolR][5], lowPredNR[symbolR][6] ])                    
                    print("symbol to remove is: ", lowPredNR[symbolR][0], lowPredNR[symbolR][1], (lowPredNR[symbolR][0]+lowPredNR[symbolR][2]), (lowPredNR[symbolR][1]+lowPredNR[symbolR][3]), lowPredNR[symbolR][4])
                    print("symbol to remove is: ", lowPredNR[symbolR], sRemove)
                    manual_checked_symbols.append([ lowPredNR[symbolR][0], lowPredNR[symbolR][1], lowPredNR[symbolR][2], lowPredNR[symbolR][3], lowPredNR[symbolR][4], lowPredNR[symbolR][5], lowPredNR[symbolR][6] ]) 

                    r=1
            else:
                print(symbolI0, "symbols were found from user defined box", sList)

        if (action==3):#add
            
            c = symbolNumber + 1

            classNo = 0

            # addedSymbol should be one of the defined classes
            for a in range(0, len(LABELS)):
                #print(LABELS[a])
                if (LABELS[a] == text):
                    classNo = a
                    print('match', LABELS[a])
                #else:
                    #print('addedSymbol class error')
                    
            addedSymbols.append([x1, y1, (x2-x1), (y2-y1), classNo, 100, c])
            manual_checked_symbols.append([x1, y1, (x2-x1), (y2-y1), classNo, 100, c])
            print("symbol added is ", x1, y1, (x2-x1), (y2-y1), classNo, 100, c)

            
            r=1

        if (action==2):#remove

            sRemove = 'empty'
            symbolI0 = 0
            sList = []
            switch = 0
            
            for p in range(0, len(lowPredNR)):##findNearestLineToRemove

                #if (LABELS[lowPredNR[p][4]] == symbolClassSelected): #match to a symbol from the chosen class   
                if (x1<=lowPredNR[p][0]<=x2) and (y1<=lowPredNR[p][1]<=y2) and (x1<=(lowPredNR[p][0] + lowPredNR[p][2])<=x2) and (y1<=(lowPredNR[p][1] + lowPredNR[p][3])<=y2):
                
                    symbolI0 = symbolI0 + 1
                    symbolR = p # index of the symbol in user defined box 
                    sList.append(p)
                    switch = 1


            for p in range(0, len(addedSymbols)):##find nearest symbol to remove, from predicted and added symbols

                #if (LABELS[addedSymbols[p][4]] == symbolClassSelected): #match to a symbol from the chosen class   
                if (x1<=addedSymbols[p][0]<=x2) and (y1<=addedSymbols[p][1]<=y2) and (x1<=(addedSymbols[p][0] + addedSymbols[p][2])<=x2) and (y1<=(addedSymbols[p][1] + addedSymbols[p][3])<=y2):
                
                    symbolI0 = symbolI0 + 1
                    symbolR = p # index of the symbol in user defined box 
                    sList.append(p)
                    switch = 2


            if (symbolI0==1): # if only 1 symbol was in the user defined box:

                if (switch==1):

                    sRemove = LABELS[lowPredNR[symbolR][4]] 
                    removedSName.append([ lowPredNR[symbolR][0], lowPredNR[symbolR][1], lowPredNR[symbolR][2], lowPredNR[symbolR][3], lowPredNR[symbolR][4], lowPredNR[symbolR][5], lowPredNR[symbolR][6] ])                    
                    print("symbol to remove is: ", lowPredNR[symbolR][0], lowPredNR[symbolR][1], (lowPredNR[symbolR][0]+lowPredNR[symbolR][2]), (lowPredNR[symbolR][1]+lowPredNR[symbolR][3]), lowPredNR[symbolR][4])
                    print("symbol to remove is: ", lowPredNR[symbolR], sRemove)
                    removedSymbols.append([ lowPredNR[symbolR][0], lowPredNR[symbolR][1], lowPredNR[symbolR][2], lowPredNR[symbolR][3], lowPredNR[symbolR][4], lowPredNR[symbolR][5], lowPredNR[symbolR][6] ]) 

                    r=1

                if (switch==2): ## remove from the added symbols 

                    print('test', symbolR, sList, addedSymbols)

                    sRemove = LABELS[addedSymbols[symbolR][4]] 
                    print("symbol to remove is: ", addedSymbols[symbolR][0], addedSymbols[symbolR][1], (addedSymbols[symbolR][0]+addedSymbols[symbolR][2]), (addedSymbols[symbolR][1]+addedSymbols[symbolR][3]), addedSymbols[symbolR][4])
                    print("symbol to remove is: ", addedSymbols[symbolR], sRemove)
                    manual_checked_symbols.remove([ addedSymbols[symbolR][0], addedSymbols[symbolR][1], addedSymbols[symbolR][2], addedSymbols[symbolR][3], addedSymbols[symbolR][4], addedSymbols[symbolR][5], addedSymbols[symbolR][6] ])   
                    addedSymbols.remove([ addedSymbols[symbolR][0], addedSymbols[symbolR][1], addedSymbols[symbolR][2], addedSymbols[symbolR][3], addedSymbols[symbolR][4], addedSymbols[symbolR][5], addedSymbols[symbolR][6] ])   

                    r=1

            else:
                print(symbolI0, "symbols were found from user defined box", sList)


        if (action==1):#adjust symbol name to one of the predefined classes

            sRemove = 'empty'
            symbolI0 = 0
            sList = []
            switchChange = 0
            
            for p in range(0, len(lowPredNR)):##findNearestLineToRemove

                #if (LABELS[lowPredNR[p][4]] == symbolClassSelected): #match to a symbol from the chosen class   
                if (x1<=lowPredNR[p][0]<=x2) and (y1<=lowPredNR[p][1]<=y2) and (x1<=(lowPredNR[p][0] + lowPredNR[p][2])<=x2) and (y1<=(lowPredNR[p][1] + lowPredNR[p][3])<=y2):
                
                    symbolI0 = symbolI0 + 1
                    symbolR = p # index of the symbol in user defined box 
                    sList.append(p)
                    switchChange = 1
                    

            # for p in range(0, len(addedSymbols)):##find nearest symbol to remove, from predicted and added symbols

            #     #if (LABELS[addedSymbols[p][4]] == symbolClassSelected): #match to a symbol from the chosen class   
            #     if (x1<=addedSymbols[p][0]<=x2) and (y1<=addedSymbols[p][1]<=y2) and (x1<=(addedSymbols[p][0] + addedSymbols[p][2])<=x2) and (y1<=(addedSymbols[p][1] + addedSymbols[p][3])<=y2):
                
            #         symbolI0 = symbolI0 + 1
            #         symbolR = p # index of the symbol in user defined box 
            #         sList.append(p)
            #         switchChange = 2



            if (symbolI0==1): # if only 1 symbol was in the user defined box:

                if (switchChange==1):

                    sRemove = LABELS[lowPredNR[symbolR][4]] 
                    #removedSName.append([ lowPredNR[symbolR][0], lowPredNR[symbolR][1], lowPredNR[symbolR][2], lowPredNR[symbolR][3], lowPredNR[symbolR][4], lowPredNR[symbolR][5], lowPredNR[symbolR][6] ])   
                     
                    print("symbol to adjust is: ", lowPredNR[symbolR][0], lowPredNR[symbolR][1], (lowPredNR[symbolR][0]+lowPredNR[symbolR][2]), (lowPredNR[symbolR][1]+lowPredNR[symbolR][3]), lowPredNR[symbolR][4])
                    print("symbol to adjust is: ", lowPredNR[symbolR], sRemove)

                    classNo = 0

                    # addedSymbol should be one of the defined classes
                    for a in range(0, len(LABELS)):
                        #print(LABELS[a])
                        if (LABELS[a] == text):
                            classNo = a
                            print('match', LABELS[a])
                    
                    changedSymbols.append([ lowPredNR[symbolR][0], lowPredNR[symbolR][1], lowPredNR[symbolR][2], lowPredNR[symbolR][3], lowPredNR[symbolR][4], lowPredNR[symbolR][5], lowPredNR[symbolR][6] ]) 
                    changedSymbolsName.append([ lowPredNR[symbolR][0], lowPredNR[symbolR][1], lowPredNR[symbolR][2], lowPredNR[symbolR][3], lowPredNR[symbolR][4], lowPredNR[symbolR][5], lowPredNR[symbolR][6], classNo ]) 
                    manual_checked_symbols.append([ lowPredNR[symbolR][0], lowPredNR[symbolR][1], lowPredNR[symbolR][2], lowPredNR[symbolR][3], classNo, lowPredNR[symbolR][5], lowPredNR[symbolR][6]]) 

                    r=1


                # if (switchChange==2):

                #     sRemove = LABELS[addedSymbols[symbolR][4]]  
                #     #removedSName.append([ boxesPID[symbolR][0], boxesPID[symbolR][1], boxesPID[symbolR][2], boxesPID[symbolR][3], boxesPID[symbolR][4], boxesPID[symbolR][5], boxesPID[symbolR][6] ])   

                #     #print('test', symbolR, sList, addedSymbols)

                #     #sRemove = LABELS[addedSymbols[symbolR][4]] 
                #     print("symbol to remove is: ", addedSymbols[symbolR][0], addedSymbols[symbolR][1], (addedSymbols[symbolR][0]+addedSymbols[symbolR][2]), (addedSymbols[symbolR][1]+addedSymbols[symbolR][3]), addedSymbols[symbolR][4])
                #     print("symbol to remove is: ", addedSymbols[symbolR], sRemove)

                #     classNo = 0

                #     # addedSymbol should be one of the defined classes
                #     for a in range(0, len(LABELS)):
                #         #print(LABELS[a])
                #         if (LABELS[a] == text):
                #             classNo = a
                #             print('match', LABELS[a])

                #     addedSymbols.append([ addedSymbols[symbolR][0], addedSymbols[symbolR][1], addedSymbols[symbolR][2], addedSymbols[symbolR][3], classNo, addedSymbols[symbolR][5], addedSymbols[symbolR][6] ])   
                #     addedSymbols.remove([ addedSymbols[symbolR][0], addedSymbols[symbolR][1], addedSymbols[symbolR][2], addedSymbols[symbolR][3], addedSymbols[symbolR][4], addedSymbols[symbolR][5], addedSymbols[symbolR][6] ])   
      
                #     r=1
            else:

                print(symbolI0, "symbols were found from user defined box", sList)




        print("after change, detected symbols: ", len(boxesPID), " removed symbols:", len(removedSymbols), "changed symbols: ", len(changedSymbols), "added symbols: ", len(addedSymbols)) 

    else:
        print('no h w')
    
    return r




## textBoxesCombine function ###################################################################################################################################
## function that will join overlapping text boxes  ###################################################################################################################################
def textBoxesCombine(boxes):
    
    listOfTextBoxesCombined = []
    overlap = 0
    
    #print("length of text boxes before combining:", len(boxes))
    #print("length of combined text boxes before:", len(listOfTextBoxesCombined))
    
    for n in range(0,len(boxes)): ##loop through the labelled boxes
        #print("box1 n", n)
        box1 = [boxes[n][0], boxes[n][1], boxes[n][2], boxes[n][3]]
        included = 0
        s = 0
        if ((boxes[n][2]-boxes[n][0])>(boxes[n][3]-boxes[n][1])):#horizontal text
            textD1 = 1
        else:
            textD1 = 0
        
        
   # for (startX, startY, endX, endY) in boxes:
        for k in range(0,len(boxes)):
            #print("box2")
            box2 = [boxes[k][0], boxes[k][1], boxes[k][2], boxes[k][3]]
            
            if ((boxes[k][2]-boxes[k][0])>(boxes[k][3]-boxes[k][1])):#horizontal text
                textD2 = 1
            else:
                textD2 = 0
    
            box1x2 = box1[2]
            box1y2 = box1[3]
            
            box2x2 = box2[2]
            box2y2 = box2[3]
            
            #coords of iou rectangle
            xLow = max(box1[0], box2[0]) ##max of box1 low or box 2 low
            xHigh = min(box1x2, box2x2)
            yLow = max(box1[1], box2[1])
            yHigh = min(box1y2, box2y2) 
            
          #  iou = 0
            
            #tempPredCentreX = boxesPID[k][0] + ((boxesPID[k][2])/2)
            #tempPredCentreY = boxesPID[k][1] + ((boxesPID[k][3])/2)
            
            if (xLow<=xHigh) or (yLow<=yHigh):
                #print("calculating overlap")
            
                areaOverlap = (xHigh-xLow) * (yHigh-yLow)
                
                #areaBox1 = (box1x2 - box1[0])*(box1y2 - box1[1])
                #areaBox2 = (box2x2 - box2[0])*(box2y2 - box2[1])
                if (box1 == box2):##the same box
                    s = s+1                        
                elif (areaOverlap>0) and (included==0) and (textD1 == textD2):
                    
                    xLow = min(box1[0], box2[0])
                    xHigh = max(box1x2, box2x2)
                    yLow = min(box1[1], box2[1])
                    yHigh = max(box1y2, box2y2) 
                    #print("foundOverlap")
                    listOfTextBoxesCombined.append([xLow, yLow, xHigh, yHigh])
                    included = 1
                    overlap = 1

        if (included ==0):##no overlaps, add the current box
            listOfTextBoxesCombined.append([boxes[n][0], boxes[n][1], boxes[n][2], boxes[n][3]])    
                    
                    
    #print("length of text boxes after combining with duplicates:", len(listOfTextBoxesCombined))     
    
    n = len(listOfTextBoxesCombined)
    
    
    listOfTextBoxesCombined.sort()
    
    listOfTextBoxesCombined = list(listOfTextBoxesCombined for listOfTextBoxesCombined,_ in itertools.groupby(listOfTextBoxesCombined))
    
    #print("length of text boxes after combining without duplicates:", len(listOfTextBoxesCombined))     
    #print("length of text boxes before and after combining", n, len(listOfTextBoxesCombined)) 
    return listOfTextBoxesCombined, overlap

## detectText function ###################################################################################################################################
## function to detect text in drawing ################################################################################################################################### 
def detectText():
    
    
    global itemNo
    global nameForCSVFile
    min_confidence = 0.5
    
    def decode_predictions(scores, geometry):
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []
    
        for y in range(0, numRows):
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]
    
            for x in range(0, numCols):
                if scoresData[x] < min_confidence:
                    continue
    
                (offsetX, offsetY) = (x * 4.0, y * 4.0)
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]
                
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)
    
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])
    
        return (rects, confidences)
    
    filenameToRead = currentPID  
    
    filenameImageA1 = currentPID[:-4] + "_A1.png"   
    
    imageA1 = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameImageA1)) 
    imageA4 = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameImageA1)) 
    
    filenameToReadOutput = currentPID[:-4] + "_detection.png"  
    
    imageOutput = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameToReadOutput))
    
    filenameToReadInput = currentPID  
    
    imageWh = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameToReadInput))
    imageTG = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameToReadInput))   
    imageText = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameToReadInput))    
    filenameImageA4 = currentPID[:-4] + "_A4.png"    
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], filenameImageA4), imageA4) 
    
    global highestXS
    #imageWh = imageWh[0:imH, 0:highestXS]
    global origH, origW
    
    print("itemNo before detect text:", itemNo)
    
    if (isCropped == 0):
        
        imageWh = imageWh[yC:(yC+ hC), xC:(xC + wC)]
    
     
    (imH, imW) = imageWh.shape[:2]     
    
    if (isCropped == 0) or ((isCropped == 1) and (imH>(0.6*origH)) and (imW>(0.6*origW))): 
       
        print("detect text in grids")
        imH = int(imH/2)
        imW = int(imW/2)
  
    else:
        print("detect text for whole image")        
        
    qH,rH = divmod(imH, 32)
    qW,rW = divmod(imW, 32)
    
    windowsize_h = qH * 32
    windowsize_w = qW * 32   


    print("origH, imH, window size h:", origH, imH, windowsize_h)
    print("origW, imW, window size w:", origW, imW, windowsize_w)
        
    textGrid = 0
    match = 0
    global textJoin
    textJoin = []
    
    global results
    results = []
    global boxesPadding
    boxesPadding =[]
    
    
    
    global nameForCSVFileText
    
    with open(os.path.join(app.config['UPLOAD_FOLDER'], nameForCSVFile), 'a', newline = '') as outfile, open(os.path.join(app.config['UPLOAD_FOLDER'], nameForCSVFileText), 'a', newline = '') as outfileText, open(os.path.join(app.config['UPLOAD_FOLDER'], nameForCSVFileTextOnly), 'a', newline = '') as outfileTextOnly:


        csvWriter = csv.writer(outfile, delimiter = ',')
        csvWriter.writerow(['Detected Text'])
        csvWriter.writerow(['diagramNo', 'item', 'item number', 'x', 'y', 'width', 'height', 'predicted text'])


        csvWriterText = csv.writer(outfileText, delimiter = ',')
        csvWriterText.writerow(['Detected Text'])
        csvWriterText.writerow(['diagramNo', 'item', 'item number', 'x', 'y', 'width', 'height', 'predicted text'])    


        csvWriterTextOnly = csv.writer(outfileTextOnly, delimiter = ',')
        csvWriterTextOnly.writerow(['Detected Text'])
        csvWriterTextOnly.writerow(['diagramNo', 'item', 'item number', 'x', 'y', 'width', 'height', 'predicted text'])    


    
    for h in range(0,imageWh.shape[0] - windowsize_h, windowsize_h):###y height y
        for w in range(0,imageWh.shape[1] - windowsize_w, windowsize_w):###x width x
         
    
            textGrid = textGrid + 1
            #print("textGrid", textGrid)
            image = imageWh[h:h+windowsize_h,w:w+windowsize_w]
            #print(h, h+windowsize_h, w, w+windowsize_w)
            textGridName = str(textGrid) + "_textPatch.png"
            
            cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], textGridName), image) ##**save the patch with the detections here 
            
            cv2.rectangle(imageTG, (w, h), ((w+windowsize_w), (h+windowsize_h)), (255,0,0), 4)

            
            orig = image.copy()
            (origH, origW) = image.shape[:2]
            
            print('origH orig W', origH, origW)


            (H, W) = image.shape[:2]

            print('H W', H, W)

            
            layerNames = [
                "feature_fusion/Conv_7/Sigmoid",
                "feature_fusion/concat_3"]
            # load the pre-trained EAST text detector
            print("starting text detection for section", textGrid)
     
            nameForNetwork = 'frozen_east_text_detection.pb'
            net = cv2.dnn.readNet(os.path.join(app.config['UPLOAD_FOLDER'], nameForNetwork))  
            
            blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                (123.68, 116.78, 103.94), swapRB=True, crop=False)
            
            start = time.time()
            net.setInput(blob)
            (scores, geometry) = net.forward(layerNames)
            end = time.time()
            
            print("time for text detection:{:.6f} seconds".format(end - start))

            (rects, confidences) = decode_predictions(scores, geometry)
            boxes = non_max_suppression(np.array(rects), probs=confidences)

            paddingY = 0.1
            paddingX = 0.12
            errors = 0

            timeT = str(int(time.time()))      
            
            #fCSV = currentPID[:-4] + "_text_" + timeT + ".csv"   
            
             #   fCSV = 'temp.csv'
            #downloadFilenameCSV = 'text.csv'
            
            
            #with open(os.path.join(app.config['UPLOAD_FOLDER'], fCSV), 'w', newline = '') as outfile:
            
            start = time.time()
            
            
            
            for (startX, startY, endX, endY) in boxes:

                    startX = int(startX)  
                    startY = int(startY)
                    endX = int(endX)
                    endY = int(endY)
                
                    dX = int((endX - startX) * paddingX)
                    dY = int((endY - startY) * paddingY)
                
                    startX = max(0, startX - dX)
                    startY = max(0, startY - dY)
                    endX = min(origW, endX + (dX * 2))
                    endY = min(origH, endY + (dY * 2))
#                        
                 
                    startX = w + startX  #for whole image
                    startY = h + startY
                    endX = w + endX
                    endY = h + endY 
                 
                    startX = startX + xC  #for whole image
                    startY = startY + yC
                    endX = endX + xC
                    endY = endY + yC                     
                
                    boxesPadding.append([startX, startY, endX, endY])                    
                
                    #results.append(((startX, startY, endX, endY), text))
                
                #global results
                #results = sorted(results, key=lambda r:r[0][1])
                
               # t = "     plp"
                #t = t.lstrip()

    print("draw boxes before combining")
    
    
    #filenameImageA4 = currentPID[:-4] + "_A4.png"

    imageA4 = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameImageA4)) #for testing
    
    
    
    for m in range(0, len(boxesPadding)):
        
        cv2.rectangle(imageA4, (boxesPadding[m][0], boxesPadding[m][1]), (boxesPadding[m][2], boxesPadding[m][3]), (0, 0, 255), 5)  
    
    overlap = 1
    n=0
    while (overlap==1):
        n = n+1
        #print("merging overlapping text boxes iteration", n)
        boxesPadding, overlap = textBoxesCombine(boxesPadding)

    print("draw boxes after combining")            
    for m in range(0, len(boxesPadding)):
                    
        cv2.rectangle(imageA4, (boxesPadding[m][0], boxesPadding[m][1]), (boxesPadding[m][2], boxesPadding[m][3]), (255, 0, 0), 2)  
    
    
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], filenameImageA4), imageA4) 

    print("starting text recognition ..")
    for (startX, startY, endX, endY) in boxesPadding:

        startX = int(startX)  #for whole image
        startY = int(startY)
        endX = int(endX)
        endY = int(endY)
    
#               dX = int((endX - startX) * paddingX)
#               dY = int((endY - startY) * paddingY)
#
#               startX = max(0, startX - dX)
#               startY = max(0, startY - dY)
#               endX = min(origW, endX + (dX * 2))
#               endY = min(origH, endY + (dY * 2))
        roi = imageText[startY:endY, startX:endX]                    
        #cv2.imwrite('roi.jpg', roi)
        
        img = Image.fromarray(roi)    
        
        config = ("-l eng --oem 1 --psm 3") 
    
        #text = pytesseract.image_to_string(Image.open(os.path.sep.join([pathToRFolder, "roi.jpg"])), config=config)

        #text = pytesseract.image_to_string(img, config=config)   

        try:        
           text = pytesseract.image_to_string(img, config=config)
        except:
            
           errors = errors +1
           filenameE = currentPID[:-4] + 'error' + str(errors) + '.png' 
           cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], filenameE), roi)
            
           continue
                        
                      
        
        results.append(((startX, startY, endX, endY), text))
    
    #global results
    results = sorted(results, key=lambda r:r[0][1])
                
                
                
    end = time.time()
             
                
    with open(os.path.join(app.config['UPLOAD_FOLDER'], nameForCSVFile), 'a', newline = '') as outfile, open(os.path.join(app.config['UPLOAD_FOLDER'], nameForCSVFileText), 'a', newline = '') as outfileText, open(os.path.join(app.config['UPLOAD_FOLDER'], nameForCSVFileTextOnly), 'a', newline = '') as outfileTextOnly:


        csvWriter = csv.writer(outfile, delimiter = ',')
        csvWriterText = csv.writer(outfileText, delimiter = ',')
        csvWriterTextOnly = csv.writer(outfileTextOnly, delimiter = ',')        


        print("number of text detections found: ", len(results))

        tNo = 0
        
        for ((startX, startY, endX, endY), text) in results:

        
            #text = "".join([c if ord(c) < 128 else "" for c in text]).strip() # ord(c) represents the unicode character, 
            text = "".join([c for c in text]).strip()  
            #text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
            #print("adjusted text: {}\n".format(text))
    
            #cv2.rectangle(imageOutput, (startX, startY), (endX, endY), (0, 0, 255), 2)        
            cv2.rectangle(imageA1, (startX, startY), (endX, endY), (255, 0, 255), 8)                
            cv2.rectangle(imageA1, (startX, startY), (endX, endY), (255, 255, 255), -1)                
            #cv2.rectangle(imageA2, (startX, startY), (endX, endY), (255, 255, 255), -1)                
            
            #cv2.putText(imageOutput, text, (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
            pid = currentPID[:-4]
            textW = endX-startX
            
            textH = endY-startY
            itemNo = itemNo + 1
            tNo = tNo + 1

            csvWriter.writerow([pid, 'text', itemNo, startX, startY, textW, textH, text])
            csvWriterText.writerow([pid, 'text', itemNo, startX, startY, textW, textH, text])
            csvWriterTextOnly.writerow([pid, 'text', tNo, startX, startY, textW, textH, text])
    
    
    filenameProcessed = currentPID[:-4] + "_detection.png"

    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], filenameProcessed), imageOutput)

    filenameImageA1 = currentPID[:-4] + "_A1.png"   ##******change the filename here #outlined rectangle
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], filenameImageA1), imageA1) ##**save the patch with the detections here     
  
    #filenameImageA2 = currentPID[:-4] + "_A2.png"   ##******change the filename here #blank rectangle
    #cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], filenameImageA2), imageA2) ##**save the patch with the detections here 

    filenameImageA4 = currentPID[:-4] + "_A4.png"   ##******change the filename here #blank rectangle
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], filenameImageA4), imageA4) ##**save the patch with the detections here 

    timeValue = str(int(time.time()))
    global filenamePIDTimeS
    
    filenamePIDTimeS = currentPID[:-4] + "_" + timeValue + "_detection.png"#save image with time
    
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], filenamePIDTimeS), imageOutput)    

    print('no of misses with text reading/encoding: ', errors)
    
    print("item no after detect text:", itemNo)
    global totalSymbolsText
    totalSymbolsText = itemNo



## drawText function ###################################################################################################################################
## function to draw onto the imageInput, boxes round the detected text  ###################################################################################################################################            
def drawText(imageInput):
    
    print("drawText")
    
    global results
    global totalSymbols
    global symbolCount
    global itemList
    
    textCounter = totalSymbols
    print("textCounter for draw text starts at..", textCounter)
    print("item count before text drawn on PID:", symbolCount)
    
    print("drawText no of results", len(results))
    
    for ((startX, startY, endX, endY), text) in results:
        #text = "".join([c if ord(c) < 128 else "" for c in text]).strip()   
        text = "".join([c for c in text]).strip()           
        textCounter = textCounter +1
        #symbolCount = symbolCount +1
        cv2.rectangle(imageInput, (startX, startY), (endX, endY), (0, 0, 255), 2)
        
        if(startY - 10 > 0):
          yT = startY - 10
        else:
          yT = 0        
        
        cv2.putText(imageInput, text, (startX, yT), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)    
            
        t= str(textCounter) 
        #t = str(symbolCount)
        #itemList.append([symbolCount, startX, startY, endX, endY, "text", text])
            
        if(startY - 30 > 0):
          yT = startY - 30
        else:
          yT = 0      
       
        drawNumber(imageInput, t, startX, yT) 
        
        
    print("item count after text drawn on PID:", symbolCount)
    #print("itemList after text drawn on PID:", len(itemList))
    print("textCounter after text drawn on PID:", textCounter)


       
## largestComponent function ###################################################################################################################################
## function to determine the largest component in the drawing  ###################################################################################################################################     
def largestComponent():
    
    print("borders being removed..")
    global xC
    global yC
    global wC
    global hC
    
    filenameToRead = currentPID  ##reads the uploaded file   
    image = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameToRead), 0)

    #ret, thresh = cv2.threshold(image, 127, 255, 0)##
  
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[:, -1]
    
    #print("stats length", len(stats))
    
    max_label = 1
    max_size = sizes[1]
    for i in range(1, nb_components):###adjust number 
        if sizes[i] > max_size:
            max_size = sizes[i]
     
            xC = stats[i,0]
            yC = stats[i,1]
            wC = stats[i,2]
            hC = stats[i,3]
            
    print("borders:", xC, yC, wC, hC)
    
    image2 = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameToRead))
    
    cv2.rectangle(image2, (xC, yC), (xC + wC, yC + hC), (255, 0, 255), 5)
    
    name = currentPID[:-4] + "_cc.png"
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], name), image2)

## uploadAll function ###################################################################################################################################
## function to process diagram ################################################################################################################################### 

@app.route('/uploadAll', methods=['POST'])
#@nocache
def uploadAll(): 

    global symbolList

    startTimer = time.time()
    
    if (isCropped == 0):
        largestComponent()

    else: #### use the size of the original image
        print('use whole image size, no border cropping')

        xC = 0
        yC = 0 
        wC = origW
        hC = origH 

        print('image details', xC, yC, wC, hC)
    
    detectSymbols()

    detectText()
    
    detectPipelines()   

    detailedSymbolsData()
 
    
    global C1, C2, C3
    C1 = 1
    C2 = 1
    C3 = 1
    
    global a,d 
    global detectNo
    filenameToReadInput = currentPID
    imageInput = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameToReadInput))
    
    print("filename to read input is: ", filenameToReadInput)
    
    filenameToReadOutputS = currentPID[:-4] + "_detection.png"  ##reads the file with the detections shown 
    
    
    
    if (C1==1):#show symbols
        
        drawSymbols(imageInput)


    if (C2==1):#show text

        drawText(imageInput)
    
    
    if (C3==1):#show pipelines 
 
        drawPipelines(imageInput)



    # ### save an image of the diagram with detected text
    # imageText = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameToReadInput))    
    # drawText(imageText)
    # imageTextname = currentPID[:-4] + "_detection_text.png"
    # cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], imageTextname), imageText) 
 




        
    createItemList()
    
    imageOutput = imageInput
    
    print("filename to read output is: ", filenameToReadOutputS)
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], filenameToReadOutputS), imageOutput)
    
    timeValue = str(int(time.time()))
    global filenamePIDTimeS                
    filenamePIDTimeS = currentPID[:-4] + "_" + timeValue + "_detection.png"#save image with time
    
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], filenamePIDTimeS), imageOutput) ###

    imageAll = currentPID[:-4] + "_detection_all.png"#save image with time
    
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], imageAll), imageOutput) ###

    
    fL = filenamePIDTimeS
    
    downloadFilename = currentPID[:-4] + "_detection.png"
    
    fCSV = nameForCSVFile
    
    downloadFilenameCSV = currentPID[:-4]  + ".csv"
    
    
    endTimer = time.time()
    
    totalT = endTimer -startTimer 
    m,s = divmod(totalT, 60)
    print("total time for processing image:", m, 'minutes', s, 'seconds')
    
    global symbol, t, pipeline
    symbol=1
    t=1
    pipeline=1
        
    fCSV = nameForCSVFile
    fCSVSymbols = nameForCSVFileSymbols
    fCSVPipelines = nameForCSVFilePipelines
    fCSVText = nameForCSVFileText
    downloadFilenameCSV = currentPID[:-4]  + ".csv"
    downloadFilenameCSVSymbols = currentPID[:-4]  + "_symbols.csv"
    downloadFilenameCSVText = currentPID[:-4]  + "_text.csv"
    downloadFilenameCSVPipelines  = currentPID[:-4]  + "_pipelines.csv"
    textOnlyFilename = currentPID[:-4]  + "_text_output.csv"
          
    count = 0
    for k in range(len(boxesPID)):
         #if (boxesPID[k][4] <= 28):
        count = count + 1
        symbolList.append(str(count) + ':' + LABELS[boxesPID[k][4]] )  

    symbolClassIndex = 0

    return render_template('p22.html', filenameL=os.path.join(app.config['OUTPUT_FOLDER'], fL), result_text=imageShown, symbolClassList = symbolClassList, sCI = symbolClassIndex, drop_list = symbolList, downloadName = downloadFilename, filenameCSV = os.path.join(app.config['UPLOAD_FOLDER'], fCSV), downloadNameCSV = downloadFilenameCSV,
                           filenameCSVSymbols = os.path.join(app.config['UPLOAD_FOLDER'], fCSVSymbols), downloadNameCSVSymbols = downloadFilenameCSVSymbols, isSymbol=symbol,
                           filenameCSVText = os.path.join(app.config['UPLOAD_FOLDER'], fCSVText), downloadNameCSVText = downloadFilenameCSVText, 
                           filenameCSVPipelines = os.path.join(app.config['UPLOAD_FOLDER'], fCSVPipelines), downloadNameCSVPipelines = downloadFilenameCSVPipelines)    
   




## si function ###################################################################################################################################
## function to process any manual changes to the netlist ################################################################################################################################### 

@app.route('/si', methods=['POST', 'GET'])
#@nocache
def si():
    
    print("si")
    n1 = request.values['n1']
    n2 = request.values['n2']
    n3 = request.values['n3']
    n4 = request.values['n4']
    n5 = request.values['n5']
    n6 = request.values['n6']
    action = request.values['n7']
    #inputText = request.values['n8']
    inputText = request.values['n10']

    global panX 
    global panY
    global sf

    panX = request.values['a1']
    panY = request.values['a2']
    sf = request.values['a5']

    print("values received:", n1, n2, n3, n4, n5, n6, action, inputText)
    
    global origW, origH
    
    im = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], currentPID))
    imH, imW = im.shape[:2]
    
    global pX, pY
    
    pX = ( int(float(n3)) /int(float(n1)) ) 
    pY = ( int(float(n4)) /int(float(n2)) ) 
    
    pX2 = ( int(float(n5)) /int(float(n1)) )  
    pY2 = ( int(float(n6)) /int(float(n2)) ) 
      
    xp1B=pX*imW
    yp1B=pY*imH
    xp2B=pX2*imW
    yp2B=pY2*imH
    
    ###convert (x1,y1) for tl, (x2,y2) for br
    
    if(xp2B>=xp1B):
        xp1 = xp1B
        xp2 = xp2B
    elif(xp2B<xp1B):
        xp1 = xp2B
        xp2 = xp1B
        
    if(yp2B>=yp1B):
        yp1 = yp1B
        yp2 = yp2B
    elif(yp2B<yp1B):
        yp1 = yp2B
        yp2 = yp1B        
    

    print("width: ", n1, " height: ", n2, "x1 y1", n3, n4, " x2 y2", n5, n6, "pX:", pX, "pY", pY, "pX2", pX2, "pY2", pY2)
    print("image width and height:", imW, imH, " xp1 yp1: ", round(xp1,0), round(yp1,0), " xp2 yp2: ", round(xp2,0), round(yp2,0), 'text', inputText, action)
    
    action = int(action)


    if (action==1):
        print("request to adjust symbol name")
        r=nearestS(xp1, yp1, xp2, yp2, action, inputText);

        if (r==0):
            print("no symbol changes")
        elif (r==1):
            print("symbol adjusted")

    if (action==2):
        print("request to remove symbol")
        r=nearestS(xp1, yp1, xp2, yp2, action, inputText);

        if (r==0):
            print("no symbol changes")
        elif (r==1):
            print("symbol removed")

    if (action==3):
        print("request to add symbol")
        r=nearestS(xp1, yp1, xp2, yp2, action, inputText);
    
        if (r==0):
            print("no symbol changes")
        elif (r==1):
            print("symbol added")
            ##updateSymbolsCSV() 


    if (action==4):
        print("request to approve symbol")
        r=nearestS(xp1, yp1, xp2, yp2, action, inputText);
    
        if (r==0):
            print("no symbol changes")
        elif (r==1):
            print("symbol approved")
            ##updateSymbolsCSV() 

    #createChangeCSV()        

    return render_template('pIndex_B.html')





## uploadTG function ###################################################################################################################################
## function that displays the P&ID with manual change options after changes  ################################################################################################################################### 
@app.route('/uploadTG', methods=['POST', 'GET'])
#@nocache
def uploadTG():##from drawing event 
    
    global symbol, t, pipeline

    global qt
 

    # find the index from the symbol type
    symbol_index = 0
    
    global C1, C2, C3
    
    print("uploadT")
    print("stp 1b",symbol,t,pipeline)    
    filenameToReadInput = currentPID
    imageInput = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameToReadInput))
    
    filenameToReadOutputS = currentPID[:-4] + "_detection.png" 
        
    if (symbol==1):#show symbols        
        drawSymbols(imageInput)

    print("total no of symbols: ", detectNo)
    print("itemCount:", symbolCount)    
    
    imageOutput = imageInput
    
    print("filename to read output is: ", filenameToReadOutputS)
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], filenameToReadOutputS), imageOutput)
    
    timeValue = str(int(time.time()))
    global filenamePIDTimeS                
    filenamePIDTimeS = currentPID[:-4] + "_" + timeValue + "_detection.png"#save image with time
    
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], filenamePIDTimeS), imageOutput) ###
    
    fL = filenamePIDTimeS
    
    downloadFilename = currentPID[:-4] + "_detection.png"

    fCSVSymbols = nameForCSVFileSymbols
    downloadFilenameCSVSymbols = currentPID[:-4]  + "_symbols.csv"
    
    print("stp 2",symbol,t,pipeline)
    print("uploadTG filename sending to view is: ", fL)

    return render_template('p20.html', filenameL=os.path.join(app.config['OUTPUT_FOLDER'], fL), symbolClassList = symbolClassList, drop_list = symbolList, sy=symbol_index, result_text=imageShown, downloadName = downloadFilename,
                            filenameCSVSymbols = os.path.join(app.config['UPLOAD_FOLDER'], fCSVSymbols), downloadNameCSVSymbols = downloadFilenameCSVSymbols,) 
                          

## drawSymbolsSelect function ###################################################################################################################################
## function to display selected symbols on the diagram  ################################################################################################################################### 
def drawSymbolsSelect(imageInput, symbolClassSelected):
    
    print("drawSymbolsSelect")

    global boxesPID
    global COLORS
    global LABELS
    global symbolCount
    global itemList

    q=0 
    selectSymbols = []
    symbolsType = []

    # list of detected symbols that match the symbolClassSelected if not removed
    for k in range(len(boxesPID)):
        if (LABELS[boxesPID[k][4]] == symbolClassSelected):
            symbolsType.append([boxesPID[k][0], boxesPID[k][1], boxesPID[k][2], boxesPID[k][3], boxesPID[k][4], boxesPID[k][5], boxesPID[k][6] ])

    selectSymbols = [x for x in symbolsType if x not in removedSymbols] # remove the removed symbols of that class
    print('there were', len(selectSymbols),'of', symbolClassSelected)
    selectSymbols = [x for x in selectSymbols if x not in changedSymbols] # remove symbols that were specified class but have changed class
    print('there were', len(selectSymbols),'of', symbolClassSelected)

    print('there were', len(selectSymbols),'of', symbolClassSelected, 'detected', len(removedSymbols), 'removed', len(changedSymbols), 'changed')

    for k in range(len(changedSymbolsName)):
        if (LABELS[changedSymbolsName[k][7]] == symbolClassSelected):
            selectSymbols.append([changedSymbolsName[k][0], changedSymbolsName[k][1], changedSymbolsName[k][2], changedSymbolsName[k][3], changedSymbolsName[k][7], changedSymbolsName[k][5], changedSymbolsName[k][6] ])

    print('there were', len(selectSymbols),'of', symbolClassSelected)

    for k in range(len(selectSymbols)):

        colorB = (0, 0, 255)
        cv2.rectangle(imageInput, (selectSymbols[k][0], selectSymbols[k][1]), (selectSymbols[k][0] + selectSymbols[k][2], selectSymbols[k][1] + selectSymbols[k][3]), colorB, 2)

        text = "{}".format(LABELS[selectSymbols[k][4]]) 
        # for b in range(len(changedSymbols)):#if it's a changed symbol, put the new symbol name instead of predicted name
        #     if ( selectSymbols[k][6] == changedSymbols[b][6] ): #symbol has been changed
        #         text = LABELS(changedSymbolsName[b][7])

        imH, imW, imD = imageInput.shape
        if((selectSymbols[k][1] + selectSymbols[k][3] + 30) <=imH):
            yT = selectSymbols[k][1] + selectSymbols[k][3] + 30
        else:
            yT = imH        
        cv2.putText(imageInput, text, (selectSymbols[k][0], yT), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colorB, 2)

        text = str(selectSymbols[k][6])
        if((selectSymbols[k][1] + selectSymbols[k][3] + 50) <= imH):
            yT = selectSymbols[k][1] + selectSymbols[k][3] + 50
        else:
            yT = imH
        drawNumber(imageInput, text, selectSymbols[k][0], yT)   
    
    print("there are addedSymbols to draw", len(addedSymbols))
    for k in range(len(addedSymbols)):
        if (LABELS[addedSymbols[k][4]] == symbolClassSelected):

            colorB = (0, 0, 255)
            cv2.rectangle(imageInput, (addedSymbols[k][0], addedSymbols[k][1]), (addedSymbols[k][0] + addedSymbols[k][2], addedSymbols[k][1] + addedSymbols[k][3]), colorB, 2)

            text = "{}".format(LABELS[addedSymbols[k][4]]) 

            imH, imW, imD = imageInput.shape
            if((addedSymbols[k][1] + addedSymbols[k][3] + 30) <=imH):
                yT = addedSymbols[k][1] + addedSymbols[k][3] + 30
            else:
                yT = imH        
            cv2.putText(imageInput, text, (addedSymbols[k][0], yT), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colorB, 2)

            text = str(addedSymbols[k][6])
            if((addedSymbols[k][1] + addedSymbols[k][3] + 50) <= imH):
                yT = addedSymbols[k][1] + addedSymbols[k][3] + 50
            else:
                yT = imH
            drawNumber(imageInput, text, addedSymbols[k][0], yT)   


## drawSymbolsAdjust function ###################################################################################################################################
## function to display adjusted symbols on the diagram  ################################################################################################################################### 
def drawSymbolsAdjust(imageInput):
    
    print("drawSymbolsAdjust")

    global boxesPID
    global COLORS
    global LABELS
    global symbolCount
    global itemList

    q=0 
    selectSymbols = []
    symbolsType = []

    # list of detected symbols that match the symbolClassSelected if not removed
    for k in range(len(boxesPID)):
        #if (LABELS[boxesPID[k][4]] == symbolClassSelected):
        symbolsType.append([boxesPID[k][0], boxesPID[k][1], boxesPID[k][2], boxesPID[k][3], boxesPID[k][4], boxesPID[k][5], boxesPID[k][6] ])

    selectSymbols = [x for x in symbolsType if x not in removedSymbols] # remove the removed symbols of that class
    print('there were', len(selectSymbols))
    selectSymbols = [x for x in selectSymbols if x not in changedSymbols] # remove symbols that were specified class but have changed class
    print('there were', len(selectSymbols))

    print('there were', len(selectSymbols), 'detected', len(removedSymbols), 'removed', len(changedSymbols), 'changed')

    for k in range(len(changedSymbolsName)):
        #if (LABELS[changedSymbolsName[k][7]] == symbolClassSelected):
        selectSymbols.append([changedSymbolsName[k][0], changedSymbolsName[k][1], changedSymbolsName[k][2], changedSymbolsName[k][3], changedSymbolsName[k][7], changedSymbolsName[k][5], changedSymbolsName[k][6] ])

    print('there were', len(selectSymbols),'of', symbolClassSelected)

    for k in range(len(selectSymbols)):

        #colorB = (0, 0, 255)

        if( selectSymbols[k][5] >= conf_al_threshold):

                 colorB = (210, 0, 150)
        else: 

                 colorB = (30, 100, 255)






        #colorB = [int(c) for c in COLORS[boxesPID[k][4]]]
        cv2.rectangle(imageInput, (selectSymbols[k][0], selectSymbols[k][1]), (selectSymbols[k][0] + selectSymbols[k][2], selectSymbols[k][1] + selectSymbols[k][3]), colorB, 2)

        #text = "{}".format(LABELS[selectSymbols[k][4]]) 
        text = "{} {:.1f}".format(LABELS[boxesPID[k][4]], boxesPID[k][5]) #labels with iou and confidence values
        # for b in range(len(changedSymbols)):#if it's a changed symbol, put the new symbol name instead of predicted name
        #     if ( selectSymbols[k][6] == changedSymbols[b][6] ): #symbol has been changed
        #         text = LABELS(changedSymbolsName[b][7])

        imH, imW, imD = imageInput.shape
        if((selectSymbols[k][1] + selectSymbols[k][3] + 30) <=imH):
            yT = selectSymbols[k][1] + selectSymbols[k][3] + 30
        else:
            yT = imH        
        cv2.putText(imageInput, text, (selectSymbols[k][0], yT), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colorB, 2)

        text = str(selectSymbols[k][6])
        if((selectSymbols[k][1] + selectSymbols[k][3] + 50) <= imH):
            yT = selectSymbols[k][1] + selectSymbols[k][3] + 50
        else:
            yT = imH
        drawNumber(imageInput, text, selectSymbols[k][0], yT)   
    
    print("there are addedSymbols to draw", len(addedSymbols))
    for k in range(len(addedSymbols)):
        #if (LABELS[addedSymbols[k][4]] == symbolClassSelected):

        #colorB = (0, 0, 255)
        colorB = [int(c) for c in COLORS[boxesPID[k][4]]]
        cv2.rectangle(imageInput, (addedSymbols[k][0], addedSymbols[k][1]), (addedSymbols[k][0] + addedSymbols[k][2], addedSymbols[k][1] + addedSymbols[k][3]), colorB, 2)

        text = "{}".format(LABELS[addedSymbols[k][4]]) 

        imH, imW, imD = imageInput.shape
        if((addedSymbols[k][1] + addedSymbols[k][3] + 30) <=imH):
            yT = addedSymbols[k][1] + addedSymbols[k][3] + 30
        else:
            yT = imH        
        cv2.putText(imageInput, text, (addedSymbols[k][0], yT), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colorB, 2)

        text = str(addedSymbols[k][6])
        if((addedSymbols[k][1] + addedSymbols[k][3] + 50) <= imH):
            yT = addedSymbols[k][1] + addedSymbols[k][3] + 50
        else:
            yT = imH
        drawNumber(imageInput, text, addedSymbols[k][0], yT)   

## drawSymbolsAdjustAl function ###################################################################################################################################
## function to display adjusted symbols on the diagram  ################################################################################################################################### 
def drawSymbolsAdjustAl(imageInput):
    
    print("drawSymbolsAdjustAl")

    global boxesPID
    global COLORS
    global LABELS
    global symbolCount
    global itemList

    lowPredNR = []
    lowPredNR = [x for x in low_pred if x not in removedSymbols] # remove the removed symbols
    lowPredNR = [x for x in lowPredNR if x not in changedSymbols] # remove the changed symbols
    lowPredNR = [x for x in lowPredNR if x not in manual_checked_symbols] # remove the approved symbols if using

    print("there are h to draw", len(high_pred))
    for k in range(len(high_pred)):
        #if (LABELS[addedSymbols[k][4]] == symbolClassSelected):

        colorB = (210, 0, 150)
        #colorB = [int(c) for c in COLORS[boxesPID[k][4]]]
        cv2.rectangle(imageInput, (high_pred[k][0], high_pred[k][1]), (high_pred[k][0] + high_pred[k][2], high_pred[k][1] + high_pred[k][3]), colorB, 2)

        text = "{} {:.1f}".format(LABELS[high_pred[k][4]], high_pred[k][5]) #labels with iou and confidence values

        imH, imW, imD = imageInput.shape
        if((high_pred[k][1] + high_pred[k][3] + 30) <=imH):
            yT = high_pred[k][1] + high_pred[k][3] + 30
        else:
            yT = imH        
        cv2.putText(imageInput, text, (high_pred[k][0], yT), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colorB, 2)

        text = str(high_pred[k][6])
        if((high_pred[k][1] + high_pred[k][3] + 50) <= imH):
            yT = high_pred[k][1] + high_pred[k][3] + 50
        else:
            yT = imH
        drawNumber(imageInput, text, high_pred[k][0], yT)   


    print("there are l to draw", len(lowPredNR))
    for k in range(len(lowPredNR)):
        #if (LABELS[addedSymbols[k][4]] == symbolClassSelected):

        colorB = (30, 100, 255)
        #colorB = [int(c) for c in COLORS[boxesPID[k][4]]]
        cv2.rectangle(imageInput, (lowPredNR[k][0], lowPredNR[k][1]), (lowPredNR[k][0] + lowPredNR[k][2], lowPredNR[k][1] + lowPredNR[k][3]), colorB, 2)

        text = "{} {:.1f}".format(LABELS[lowPredNR[k][4]], lowPredNR[k][5]) #labels with iou and confidence values

        imH, imW, imD = imageInput.shape
        if((lowPredNR[k][1] + lowPredNR[k][3] + 30) <=imH):
            yT = lowPredNR[k][1] + lowPredNR[k][3] + 30
        else:
            yT = imH        
        cv2.putText(imageInput, text, (lowPredNR[k][0], yT), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colorB, 2)

        text = str(lowPredNR[k][6])
        if((lowPredNR[k][1] + lowPredNR[k][3] + 50) <= imH):
            yT = lowPredNR[k][1] + lowPredNR[k][3] + 50
        else:
            yT = imH
        drawNumber(imageInput, text, lowPredNR[k][0], yT)   


    print("there are m to draw", len(manual_checked_symbols))
    for k in range(len(manual_checked_symbols)):
        #if (LABELS[addedSymbols[k][4]] == symbolClassSelected):

        colorB = (255, 0, 0)
        #colorB = [int(c) for c in COLORS[boxesPID[k][4]]]
        cv2.rectangle(imageInput, (manual_checked_symbols[k][0], manual_checked_symbols[k][1]), (manual_checked_symbols[k][0] + manual_checked_symbols[k][2], manual_checked_symbols[k][1] + manual_checked_symbols[k][3]), colorB, 2)

        text = "{} {:.1f}".format(LABELS[manual_checked_symbols[k][4]], manual_checked_symbols[k][5]) #labels with iou and confidence values

        imH, imW, imD = imageInput.shape
        if((manual_checked_symbols[k][1] + manual_checked_symbols[k][3] + 30) <=imH):
            yT = manual_checked_symbols[k][1] + manual_checked_symbols[k][3] + 30
        else:
            yT = imH        
        cv2.putText(imageInput, text, (manual_checked_symbols[k][0], yT), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colorB, 2)

        text = str(manual_checked_symbols[k][6])
        if((manual_checked_symbols[k][1] + manual_checked_symbols[k][3] + 50) <= imH):
            yT = manual_checked_symbols[k][1] + manual_checked_symbols[k][3] + 50
        else:
            yT = imH
        drawNumber(imageInput, text, manual_checked_symbols[k][0], yT)   






## createTrainingFile function ###################################################################################################################################
## function to create the training json file  ################################################################################################################################### 
def createTrainingFile():
    
    print("createTrainingFile")

    global boxesPID
    global COLORS
    global LABELS
    global symbolCount
    global itemList

    q=0 
    selectSymbols = []
    symbolsType = []

    jsonName = currentPID[:-4] + "_Annotations_AL.json"
    jsonList=[]

    for k in range(len(removedSymbols)):
        print('removed', LABELS[removedSymbols[k][4]])

    selectSymbolsJson = []

    selectSymbolsJson = [x for x in boxesPID if x not in removedSymbols] #all the symbols that havent been removed

    for k in range(len(selectSymbolsJson)):

        name = LABELS[selectSymbolsJson[k][4]]

        for b in range(len(changedSymbols)):#if it's a changed symbol, put the new symbol name instead of predicted name
            if ( selectSymbolsJson[k][6] == changedSymbols[b][6] ):
                name = LABELS[changedSymbolsName[b][7]]

        jsonList.append({'class':name,"height":selectSymbolsJson[k][3], "type":'rect', "width":selectSymbolsJson[k][2], "x":selectSymbolsJson[k][0], "y":selectSymbolsJson[k][1]})
 
    # if symbol has been added 
    for k in range(len(addedSymbols)):

        jsonList.append({'class':LABELS[addedSymbols[k][4]],"height":addedSymbols[k][3], "type":'rect', "width":addedSymbols[k][2], "x":addedSymbols[k][0], "y":addedSymbols[k][1]})
 

    g = {"annotations":jsonList,
         "class": "image",
         "filename": currentPID}

    f = []
    f.append(g)

    with open(os.path.join(app.config['UPLOAD_FOLDER'], jsonName), 'w') as outfile:
        json.dump(f, outfile, indent=4)

    return




## createTrainingFileAl function ###################################################################################################################################
## function to create the training json file  ################################################################################################################################### 
def createTrainingFileAl():
    
    print("createTrainingFileAl")

    global boxesPID
    global COLORS
    global LABELS
    global symbolCount
    global itemList

    q=0 
    selectSymbols = []
    symbolsType = []

    jsonName = currentPID[:-4] + "_Annotations_AL.json"
    jsonList=[]

    lowPredNR = []
    lowPredNR = [x for x in low_pred if x not in removedSymbols] # remove the removed symbols
    lowPredNR = [x for x in lowPredNR if x not in changedSymbols] # remove the changed symbols
    lowPredNR = [x for x in lowPredNR if x not in manual_checked_symbols] # remove the approved symbols if using


    ######  
    for k in range(len(high_pred)):

        jsonList.append({'class':LABELS[high_pred[k][4]],"height":high_pred[k][3], "type":'rect', "width":high_pred[k][2], "x":high_pred[k][0], "y":high_pred[k][1]})


    ######
    for k in range(len(manual_checked_symbols)):

        jsonList.append({'class':LABELS[manual_checked_symbols[k][4]],"height":manual_checked_symbols[k][3], "type":'rect', "width":manual_checked_symbols[k][2], "x":manual_checked_symbols[k][0], "y":manual_checked_symbols[k][1]})
 

    ######
    for k in range(len(lowPredNR)):

        jsonList.append({'class':LABELS[lowPredNR[k][4]],"height":lowPredNR[k][3], "type":'rect', "width":lowPredNR[k][2], "x":lowPredNR[k][0], "y":lowPredNR[k][1]})


    g = {"annotations":jsonList,
         "class": "image",
         "filename": currentPID}

    f = []
    f.append(g)

    with open(os.path.join(app.config['UPLOAD_FOLDER'], jsonName), 'w') as outfile:
        json.dump(f, outfile, indent=4)

    return

## uploadTG_3 function ###################################################################################################################################
## function  ################################################################################################################################### 
@app.route('/uploadTG_3', methods=['POST', 'GET'])
#@nocache
def uploadTG_3():## chosen specific class of symbols
    
    global symbol, t, pipeline

    global symbolClassSelected
    global symbolClassIndex

    symbolClassSelected =  request.form['select_symbol_class']

    print("value from the dropdown form: ", symbolClassSelected)

    # find the index from the symbol type
    symbolClassIndex = symbolClassList.index(symbolClassSelected)

    # find the index from the symbol type
    symbol_index = 0
    
    global C1, C2, C3
    
    print("uploadTG")
        
    filenameToReadInput = currentPID
    imageInput = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameToReadInput))

    filenameSelect = currentPID[:-4] + "_detection_select.png"  ## reads the file with the detections shown 
    
    #drawSymbolsSelect(imageInput, symbolClassSelected) ## draw symbols of specified class 

    drawSymbolsAdjustAl(imageInput) ## draw symbols of all classes
    createTrainingFileAl()

    imageOutput = imageInput
    
    print("filename to read output is: ", filenameSelect)
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], filenameSelect), imageOutput)
    
    timeValue = str(int(time.time()))
    global filenamePIDTimeSelect                
    filenamePIDTimeSelect = currentPID[:-4] + "_" + timeValue + "_detection_select.png"#save image with time
    
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], filenamePIDTimeSelect), imageOutput) ###

    fL = filenamePIDTimeSelect

    fCSVSymbols = nameForCSVFileSymbols
    downloadFilenameCSVSymbols = currentPID[:-4]  + "_symbols.csv"

    
    print("stp 2",symbol,t,pipeline)
    print("uploadTG filename sending to view is: ", fL)

    symbolListShow = [] # list to hold symbols of chosen class not yet reviewed

    for i in symbolList:
        if symbolClassSelected in i:
            symbolListShow.append(i)

    no_detect = 0
    if (len(symbolListShow) == 0):
        no_detect = 0
    else:
        no_detect = 1 

    symbolClassList2 = []
    symbolClassList2 = [ x for x in symbolClassList if ( x != symbolClassSelected ) ]

    panXS = 0
    panYS = 0
    sfS = 1

    jsonName = currentPID[:-4] + "_Annotations_AL.json"

    return render_template('p70.html', filenameTraining = os.path.join(app.config['UPLOAD_FOLDER'], jsonName), downloadNameImageTraining = currentPID, 
                            imageTraining = os.path.join(app.config['UPLOAD_FOLDER'], currentPID), downloadNameTraining = jsonName, pX = panXS, pY = panYS, scale = sfS, filenameL=os.path.join(app.config['OUTPUT_FOLDER'], fL), sym = symbolClassSelected, noDetect = no_detect, symbolClassList2 = symbolClassList2, symbolClassList = symbolClassList, sCI = symbolClassIndex, drop_list = symbolListShow, sy=symbol_index, result_text=imageShown,
                            filenameCSVSymbols = os.path.join(app.config['UPLOAD_FOLDER'], fCSVSymbols), downloadNameCSVSymbols = downloadFilenameCSVSymbols) 
                          


## uploadTG_3_all function ###################################################################################################################################
## function to manually review all predictions ################################################################################################################################### 
@app.route('/uploadTG_3_all', methods=['POST', 'GET'])
#@nocache
def uploadTG_3_all():## chosen specific class of symbols
    
    global symbol, t, pipeline

    global symbolClassSelected
    global symbolClassIndex

    # symbolClassSelected =  request.form['select_symbol_class']
    # print("value from the dropdown form: ", symbolClassSelected)
    # find the index from the symbol type
    #symbolClassIndex = symbolClassList.index(symbolClassSelected)

    # find the index from the symbol type
    symbol_index = 0
    
    global C1, C2, C3
    
    print("uploadTG_3_all")
        
    filenameToReadInput = currentPID
    imageInput = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameToReadInput))

    filenameSelect = currentPID[:-4] + "_detection_select.png"  ## reads the file with the detections shown 
    
    #drawSymbolsSelect(imageInput, symbolClassSelected) ## draw symbols of specified class 
    drawSymbolsAdjustAl(imageInput) ## draw symbols of all classes
    createTrainingFileAl()

    drawText(imageInput) 
 
    
    drawPipelines(imageInput)

    imageOutput = imageInput
    
    print("filename to read output is: ", filenameSelect)
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], filenameSelect), imageOutput)
    
    timeValue = str(int(time.time()))
    global filenamePIDTimeSelect                
    filenamePIDTimeSelect = currentPID[:-4] + "_" + timeValue + "_detection_select.png"#save image with time
    
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], filenamePIDTimeSelect), imageOutput) ###

    fL = filenamePIDTimeSelect

    fCSVSymbols = nameForCSVFileSymbols
    downloadFilenameCSVSymbols = currentPID[:-4]  + "_symbols.csv"

    
    #print("stp 2",symbol,t,pipeline)
    print("uploadTG filename sending to view is: ", fL)

    symbolListShow = [] # list to hold symbols of chosen class not yet reviewed

    # for i in symbolList:
    #     if symbolClassSelected in i:
    #         symbolListShow.append(i)

    no_detect = 0
    if (len(symbolListShow) == 0):
        no_detect = 0
    else:
        no_detect = 1 

    symbolClassList2 = []
    #symbolClassList2 = [ x for x in symbolClassList if ( x != symbolClassSelected ) ]
    symbolClassList2 = symbolClassList[1:]

    panXS = 0
    panYS = 0
    sfS = 1

    jsonName = currentPID[:-4] + "_Annotations_AL.json"

    return render_template('p71.html', filenameTraining = os.path.join(app.config['UPLOAD_FOLDER'], jsonName), downloadNameImageTraining = currentPID, 
                            imageTraining = os.path.join(app.config['UPLOAD_FOLDER'], currentPID), downloadNameTraining = jsonName, pX = panXS, pY = panYS, scale = sfS, 
                            filenameL=os.path.join(app.config['OUTPUT_FOLDER'], fL), sym = symbolClassSelected, noDetect = no_detect, symbolClassList2 = symbolClassList2, 
                            symbolClassList = symbolClassList, sCI = symbolClassIndex, drop_list = symbolListShow, 
                            sy=symbol_index,
                            result_text=imageShown,
                            filenameCSVSymbols = os.path.join(app.config['UPLOAD_FOLDER'], fCSVSymbols), downloadNameCSVSymbols = downloadFilenameCSVSymbols) 
                          

## uploadTG_3 function ###################################################################################################################################
## function  ################################################################################################################################### 
@app.route('/uploadTG_3_U', methods=['POST', 'GET'])
#@nocache
def uploadTG_3_U():## chosen specific class of symbols_updated
    
    global symbol, t, pipeline

    global symbolClassSelected
    global symbolClassIndex

    symbol_index = 0
    
    global C1, C2, C3
    
    print("uploadTG_3_U")

    filenameToReadInput = currentPID
    imageInput = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameToReadInput))

    filenameSelect = currentPID[:-4] + "_detection_select.png"  ## reads the file with the detections shown 
    
    #drawSymbolsSelect(imageInput, symbolClassSelected) ## draw symbols of specified class 
    drawSymbolsAdjustAl(imageInput) ## draw symbols of all classes
    createTrainingFileAl()

    drawText(imageInput) 

    
    drawPipelines(imageInput)

    imageOutput = imageInput
    
    print("filename to read output is: ", filenameSelect)
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], filenameSelect), imageOutput)
    
    timeValue = str(int(time.time()))
    global filenamePIDTimeSelect                
    filenamePIDTimeSelect = currentPID[:-4] + "_" + timeValue + "_detection_select.png"#save image with time
    
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], filenamePIDTimeSelect), imageOutput) ###

    fL = filenamePIDTimeSelect

    fCSVSymbols = nameForCSVFileSymbols
    downloadFilenameCSVSymbols = currentPID[:-4]  + "_symbols.csv"

    
    print("stp 2",symbol,t,pipeline)
    print("uploadTG_3_U filename sending to view is:",  fL, symbolRemove)

    if (symbolRemove != -1):
        symbolList.remove(symbolRemove)


    symbolListShow = []

    # for i in symbolList:
    #     if symbolClassSelected in i:
    #         symbolListShow.append(i)


    no_detect = 0
    if (len(symbolListShow) == 0):
        no_detect = 5
    else:
        no_detect = 1 

    symbolClassList2 = []
    #symbolClassList2 = [ x for x in symbolClassList if ( x != symbolClassSelected ) ]
    symbolClassList2 = symbolClassList[1:]
    jsonName = currentPID[:-4] + "_Annotations_AL.json"

    return render_template('p71.html', filenameTraining = os.path.join(app.config['UPLOAD_FOLDER'], jsonName), downloadNameImageTraining = currentPID, 
                            imageTraining = os.path.join(app.config['UPLOAD_FOLDER'], currentPID), downloadNameTraining = jsonName, pX = panX, pY = panY, scale = sf, filenameL=os.path.join(app.config['OUTPUT_FOLDER'], fL), 
                            sym = symbolClassSelected, noDetect = no_detect, symbolClassList2 = symbolClassList2, symbolClassList = symbolClassList, sCI = symbolClassIndex, drop_list = symbolListShow, sy=symbol_index, result_text=imageShown,
                            filenameCSVSymbols = os.path.join(app.config['UPLOAD_FOLDER'], fCSVSymbols), downloadNameCSVSymbols = downloadFilenameCSVSymbols) 
                          



# ## uploadTG function ###################################################################################################################################
# ## function that displays the P&ID with manual change options after changes  ################################################################################################################################### 
# @app.route('/uploadTG_2', methods=['POST', 'GET'])
# #@nocache
# def uploadTG_2():##from drawing event 
    
#     global symbol, t, pipeline
#     global qt
#     global symbolList
#     global symbolRemove
 
#     print("stp 1",symbol,t,pipeline)

#     symbolRemove = request.form['select_symbol']
#     print("value from the dropdown form: ", symbolRemove)

#     # find the index from the symbol type
#     symbol_index = symbolList.index(symbolRemove)

#     print("uploadTG_2")
        
#     filenameToReadInput = currentPID
#     imageInput = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], filenameToReadInput))

#     filenameSelect = currentPID[:-4] + "_detection_select.png"  
    
#     drawSymbolsSelect(imageInput, symbolClassSelected) ## draw symbols of specified class 
#     createTrainingFile()

#     imageOutput = imageInput
    
#     print("filename to read output is: ", filenameSelect)
#     cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], filenameSelect), imageOutput)
    
#     timeValue = str(int(time.time()))
#     global filenamePIDTimeSelect                
#     filenamePIDTimeSelect = currentPID[:-4] + "_" + timeValue + "_detection_select.png"#save image with time
    
#     cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], filenamePIDTimeSelect), imageOutput) ###

#     fL = filenamePIDTimeSelect

#     fCSVSymbols = nameForCSVFileSymbols
#     downloadFilenameCSVSymbols = currentPID[:-4]  + "_symbols.csv"

#     symbolListShow = [] # list to hold symbols of chosen class not yet reviewed

#     for i in symbolList:
#         if symbolClassSelected in i:
#             symbolListShow.append(i)

#     # find the index from the symbol type
#     symbol_index = symbolListShow.index(symbolRemove)

#     symbolListShow2 = []
#     symbolListShow2 = [ x for x in symbolListShow if (x != symbolRemove) ]

#     jsonName = currentPID[:-4] + "_Annotations_AL.json"

#     return render_template('p40.html', filenameTraining = os.path.join(app.config['UPLOAD_FOLDER'], jsonName), downloadNameTraining = jsonName, filenameL=os.path.join(app.config['OUTPUT_FOLDER'], fL), drop_list_2 = symbolListShow2, drop_list = symbolListShow, sy = symbol_index, symbolClassList = symbolClassList, sCI = symbolClassIndex, result_text=imageShown,
#                             filenameCSVSymbols = os.path.join(app.config['UPLOAD_FOLDER'], fCSVSymbols), downloadNameCSVSymbols = downloadFilenameCSVSymbols,) 
                          


## uploadL function ###################################################################################################################################
## function that will show either the drawing that was uploaded or 'select a P&ID' message on the index page ################################################################################################################################### 
@app.route('/uploadL', methods=['POST'])
#@nocache
def uploadL(): #upload whole image
    
    global origH, origW
    
    file = request.files['file']
    if file and allowed_file(file.filename):
        
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))   
        
        global currentPID
        currentPID = filename
        
        
        global imageShown
        imageShown = "Diagram Name: " + currentPID[:-4]
        
        timeValue = str(int(time.time()))      
    
        filenameTime = filename[:-4] + "_" + timeValue + ".png"
    
        fL = filenameTime
        
        imageTime = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        origH, origW = imageTime.shape[:2]
        
        print("original height and width of diagram: ", origH, origW)
        
        currentFilename = filename[:-4] + "_detection.png"
        
        imageCurrent = imageTime
        
        cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], currentFilename), imageCurrent)#output for detectiuons
        
        currentFileUnprocessed = imageTime
        
        cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], filename), currentFileUnprocessed)#uploaded file copy
        
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filenameTime), imageTime) 
        
        #adaptive_resize(os.path.join(app.config['UPLOAD_FOLDER'], filenameTime))
        
        #fS = filenameTime[:-4] + "_resized.png"
        
        downloadFilename = currentPID[:-4] + "_detection.png"
        
        
        timeValue = str(int(time.time()))      
        
        global nameForCSVFile
        nameForCSVFile = currentPID[:-4] + ".csv"
        pidNo = currentPID[:-4]
        with open(os.path.join(app.config['UPLOAD_FOLDER'], nameForCSVFile), 'w', newline = '') as outfile:
            csvWriter = csv.writer(outfile, delimiter = ',')
            #csvWriter.writerow(['pidNo', 'Labelled Class', 'x', 'y', 'width', 'height', 'Predicted Class', 'x', 'y', 'width', 'height', 'IOU', 'matching class'])
            csvWriter.writerow(['output csv file for:', pidNo])

        global nameForCSVFileSymbols
        nameForCSVFileSymbols = currentPID[:-4] + "_Symbols.csv"
        with open(os.path.join(app.config['UPLOAD_FOLDER'], nameForCSVFileSymbols), 'w', newline = '') as outfile:
            csvWriter = csv.writer(outfile, delimiter = ',')
            #csvWriter.writerow(['pidNo', 'Labelled Class', 'x', 'y', 'width', 'height', 'Predicted Class', 'x', 'y', 'width', 'height', 'IOU', 'matching class'])
            csvWriter.writerow(['output symbols csv file for:', pidNo])


        global nameForCSVFileSymbols_c
        nameForCSVFileSymbols_c = currentPID[:-4] + "_Symbols_confidence.csv"
        with open(os.path.join(app.config['UPLOAD_FOLDER'], nameForCSVFileSymbols_c), 'w', newline = '') as outfile:
            csvWriter = csv.writer(outfile, delimiter = ',')
            #csvWriter.writerow(['pidNo', 'Labelled Class', 'x', 'y', 'width', 'height', 'Predicted Class', 'x', 'y', 'width', 'height', 'IOU', 'matching class'])
            csvWriter.writerow(['output symbols csv file for:', pidNo])

        global nameForCSVFileTextOnly
        nameForCSVFileTextOnly = currentPID[:-4] + "_Text_output.csv"
        with open(os.path.join(app.config['UPLOAD_FOLDER'], nameForCSVFileTextOnly), 'w', newline = '') as outfile:
            csvWriter = csv.writer(outfile, delimiter = ',')
            #csvWriter.writerow(['pidNo', 'Labelled Class', 'x', 'y', 'width', 'height', 'Predicted Class', 'x', 'y', 'width', 'height', 'IOU', 'matching class'])
            csvWriter.writerow(['output text only csv file for:', pidNo])

        global nameForCSVFileText
        nameForCSVFileText = currentPID[:-4] + "_Text.csv"
        with open(os.path.join(app.config['UPLOAD_FOLDER'], nameForCSVFileText), 'w', newline = '') as outfile:
            csvWriter = csv.writer(outfile, delimiter = ',')
            #csvWriter.writerow(['pidNo', 'Labelled Class', 'x', 'y', 'width', 'height', 'Predicted Class', 'x', 'y', 'width', 'height', 'IOU', 'matching class'])
            csvWriter.writerow(['output text csv file for:', pidNo])
            
        global nameForCSVFilePipelines
        nameForCSVFilePipelines = currentPID[:-4] + "_Pipelines.csv"
        with open(os.path.join(app.config['UPLOAD_FOLDER'], nameForCSVFilePipelines), 'w', newline = '') as outfile:
            csvWriter = csv.writer(outfile, delimiter = ',')
            #csvWriter.writerow(['pidNo', 'Labelled Class', 'x', 'y', 'width', 'height', 'Predicted Class', 'x', 'y', 'width', 'height', 'IOU', 'matching class'])
            csvWriter.writerow(['output pipelines csv file for:', pidNo])
        

        global nameForChangeCSV
        nameForChangeCSV = currentPID[:-4] + "_Symbol_changes.csv"
        with open(os.path.join(app.config['UPLOAD_FOLDER'], nameForChangeCSV), 'w', newline = '') as outfileC:
            csvWriterChanges = csv.writer(outfileC, delimiter = ',')
            csvWriterChanges.writerow(['Alterations to the predicted symbols for diagram: ', currentPID, 'height:', origH, 'width:', origW])

        jsonName = currentPID[:-4] + "_Annotations_AL.json"
        jsonList = []
        g = {"annotations":jsonList,
             "class": "image",
             "filename": currentPID}
        f = []
        f.append(g)
        
        with open(os.path.join(app.config['UPLOAD_FOLDER'], jsonName), 'w') as outfile:
            json.dump(f, outfile, indent=4)

        return render_template('p11_crop_t2_4_1_B.html', filenameL=os.path.join(app.config['UPLOAD_FOLDER'], fL), result_text=imageShown, downloadName = downloadFilename)




    else:
        errorMsg1 = "Click on 'Choose File' to Select a Diagram"
        return render_template('pIndex_B.html', result_text=errorMsg1)
    
    

# uploaded_file function ###################################################################################################################################
# function to return uploaded file ################################################################################################################################### 
@app.route('/uploads/<filename>')
#@nocache
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

## output_file function ###################################################################################################################################
## function to return output file  - not used ################################################################################################################################### 
@app.route('/output/<filename>')
#@nocache
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'],
                               filename)
    

## resultShow function ###################################################################################################################################
## function to return output file  - not used ###################################################################################################################################  
@app.route('/resultShow/<filename>')
#@nocache
def resultShow(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'],
                               filename)
    
## downloadA function ###################################################################################################################################
## function not used ###################################################################################################################################   
@app.route('/downloadA', methods=['POST', 'GET'])
#@nocache
def downloadA():
    
    #return ""
    filename = currentPID[:-4] + "_detection.png"
    
    print("filename")
    
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

## root_file function ###################################################################################################################################
## function not used ################################################################################################################################### 
@app.route('/<filename>')
#@nocache
def root_file(filename):
    return send_from_directory('./', filename)


## Flask settings #######################################################################################################################
if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=int("9093"),
        debug=False
    )
