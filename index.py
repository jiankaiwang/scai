# -*- coding: utf-8 -*-
"""
@author: JianKai Wang (https://jiankaiwang.no-ip.biz/)
"""

from flask import Flask, render_template, request, jsonify, send_file
import io
import os
import hashlib
import datetime
import random
import json
import codecs
import subprocess
import platform
import argparse

app = Flask(__name__)
SESSPATH = os.path.join('.','session')
__randomElement = [item for item in '0123456789abcdefghijklmnopqrstuvwxyz']
RUNNING_MODE = "runtime"
OD_POOL_PATH = os.path.join('.','pooling','object_detection')
IC_POOL_PATH = os.path.join('.','pooling','image_classification')

###############################################################################
# Pre-defined Functions
###############################################################################  
def generate_keypair():
    now = datetime.datetime.now()
    random.seed(now)
    ymd = now.strftime('%Y%m%d')
    ymdhms = now.strftime('%Y/%m/%d %H:%M:%S')
    random.shuffle(__randomElement)
    salt = ''.join(__randomElement[0:8])
    s = hashlib.sha1(salt.encode('utf-8'))
    sessionName = "d-{}-{}".format(ymd,s.hexdigest())
    k = hashlib.sha1(("{}{}".format(ymdhms,salt)).encode('utf-8'))
    return sessionName, ymdhms, k.hexdigest()
    
def retUploadMsg(status, filename, datetime, sessname, sesskey):
    return {"status": status \
            , "filename": filename \
            , "datetime": datetime \
            , "sessname": sessname \
            , "sesskey": sesskey}
    
def splitFilename(filename):
    filename, file_extension = os.path.splitext(filename)
    return filename, file_extension
    
# param:
# |- state: [initialization, processing, complete, failure]  
# |- job: {x | image_classification, object_detection }
# |- key: access key must be the same
# |- input_image: the upload image
# |- message: all output message
# |- result: if state is complete, the result would show as [{class:"",score:0},{}]
# |- result_image: if state is complete, 
#                  the result_image would be a image with object labeling
def createProcessContent(processFilePath, job, inputImageName, access_key):
    try:
        with codecs.open(\
            os.path.join(processFilePath, "process.json"), "w", "utf-8") as fout:
            fout.write(json.dumps({\
                "state": "initialization" \
                , "job": job \
                , "key": access_key \
                , "input_image": inputImageName \
                , "message": "" \
                , "result" : "" \
                , "result_image": "" }))
        return 0
    except:
        return 1
        
def readProcessContent(processFilePath):
    try:
        tmpCnt = ""
        with codecs.open(\
            os.path.join(processFilePath, "process.json"), "r", "utf-8") as fin:
            for line in fin:
                tmpCnt += line.strip()
        return tmpCnt
    except:
        return ""        

def writeBackProcess(processFilePath, content):
    with codecs.open(\
        os.path.join(processFilePath, "process.json"), "w", "utf-8") as fout:
        fout.write(content)

def __write_init_status(sessPath, sessName):
    allContent = readProcessContent(os.path.join(sessPath, sessName))
    allContent = json.loads(allContent)
    allContent['state'] = "processing"
    allContent['message'] += \
        "AI recognition starts on {}.".format(datetime.datetime.now().strftime('%H:%M:%S'))
    writeBackProcess(os.path.join(sessPath, sessName), json.dumps(allContent))        
    
def __write_init_failure_status(sessPath, sessName):
    allContent = readProcessContent(os.path.join(sessPath, sessName))
    allContent = json.loads(allContent)
    allContent['state'] = "failure"
    allContent['message'] += "Session name {} can not start AI.".format(sessName)
    writeBackProcess(os.path.join(sessPath, sessName), json.dumps(allContent))
        
def __runOnBackground(sessPath, sessName, bashCommand):
    try:
        plat = platform.system()[0:3].lower()
        if plat == "win":
            process = subprocess.Popen(\
                    bashCommand.split() \
                    , stdin=None \
                    , stdout=None \
                    , stderr=None \
                    , close_fds=True)
        else:
            process = subprocess.Popen(\
                    bashCommand.split() \
                    , stdin=None \
                    , stdout=None \
                    , stderr=None \
                    , close_fds=True)      
        __write_init_status(sessPath, sessName)
    except Exception as e:
        print("Session name {} can not start AI.".format(sessName))
        __write_init_failure_status(sessPath, sessName)
        
def __deploy_task_to_pool_path(poolPath, sessPath, sessName):
    targetPath = os.path.join(poolPath, sessName)
    try:
        with codecs.open(targetPath, 'w', 'utf-8') as fout:
            fout.write('')
        __write_init_status(sessPath, sessName)
    except:
        print("Session name {} can not start AI.".format(sessName))
        __write_init_failure_status(sessPath, sessName)
        
def startImageClassification(sessPath, sessName):
    global RUNNING_MODE, IC_POOL_PATH
    if RUNNING_MODE == "job":
        bashCommand = "python label_image.py --runningmode={} --sessname={}"\
            .format(RUNNING_MODE, sessName)  
        __runOnBackground(sessPath, sessName, bashCommand)
    elif RUNNING_MODE == "runtime":
        __deploy_task_to_pool_path(IC_POOL_PATH, sessPath, sessName)
        
def startObjectDetection(sessPath, sessName):   
    global RUNNING_MODE, OD_POOL_PATH
    if RUNNING_MODE == "job":    
        bashCommand = "python object_detection.py --runningmode={} --sessname={}"\
            .format(RUNNING_MODE, sessName)
        __runOnBackground(sessPath, sessName, bashCommand)
    elif RUNNING_MODE == "runtime":
        __deploy_task_to_pool_path(OD_POOL_PATH, sessPath, sessName)        
        
def __initialize_session(task_type):
    if request.method == 'POST' and 'photo' in request.files:
        sessionFolder, ymdhms, key = generate_keypair()
        targetPath = os.path.join(SESSPATH, sessionFolder)
        if not os.path.isdir(targetPath):
            os.mkdir(targetPath)
        try:
            # upload image
            photo = request.files['photo']
            photo.save(os.path.join(targetPath, photo.filename))
            filename, file_extension = splitFilename(photo.filename)
            assert (['.jpg','.jpeg','.png']).index(file_extension.lower()) > -1
                    
            # prepare the ai environment
            os.rename(os.path.join(targetPath, photo.filename), \
                      os.path.join(targetPath, 'input' + file_extension))
            
            processFile = createProcessContent(\
                        targetPath, task_type, 'input' + file_extension, key)
            if processFile == 1:
                raise ValueError("Process File can not be created.")
                    
            # start the backend process to ai recognition
            if task_type == "image_classification":
                startImageClassification(SESSPATH, sessionFolder)
            elif task_type == "object_detection":
                startObjectDetection(SESSPATH, sessionFolder)
            
            return jsonify(retUploadMsg(\
                    "success", photo.filename, ymdhms, sessionFolder, key))
        except AssertionError:
            return jsonify(retUploadMsg(\
                    "error", "", "", "not supported file format (jpg, png)", ""))
        except ValueError:
            return jsonify(retUploadMsg(\
                    "error", "", "", "parsing process file is error", ""))
        except:
            return jsonify(retUploadMsg(\
                    "error", "", ymdhms, "upload process was error", ""))
    elif request.method == 'GET':
        return jsonify(retUploadMsg(\
                "error", "", "", "please follow the api instruction", ""))
    else:
        return jsonify(retUploadMsg(\
                "error", "", "", "please follow the api instruction", ""))    
        
# tag_in_process: {x | 'input_image','result_image'}
def __output_img(tag_in_process, job):
    allArgs = dict(request.args)
    imgPath = ""
    available = False
    preloadres = ""
    if "task" in allArgs.keys() or "key" in allArgs.keys():
        taskPath = os.path.join(SESSPATH, allArgs["task"][0])
        if os.path.isdir(taskPath):
            preloadres = json.loads(readProcessContent(taskPath))
            processKey = preloadres['key']
            if processKey == allArgs["key"][0] and preloadres['job'] == job:
                imgPath = os.path.join(".","session",allArgs["task"][0],preloadres[tag_in_process])
                available = True
    if not available:
        imgPath = os.path.join(".","usr","task404.jpg")
    with open(imgPath, "rb") as image_file:
        return send_file(io.BytesIO(image_file.read()),mimetype='image/jpeg')

def __response_res(job):        
    allArgs = dict(request.args)
    available = False
    preloadres = ""
    if "task" in allArgs.keys() or "key" in allArgs.keys():
        taskPath = os.path.join(SESSPATH, allArgs["task"][0])
        if os.path.isdir(taskPath):
            preloadres = json.loads(readProcessContent(taskPath))
            processKey = preloadres['key']
            if processKey == allArgs["key"][0] and preloadres['job'] == job:
                available = True
    if available:
        for notshown in ['key','input_image']:
            del preloadres[notshown]
        return jsonify(preloadres)
    else:
        return jsonify(retUploadMsg(\
                    "error", "", "", "no available task, key or job", ""))          


###############################################################################
# API Instruction and Homepage
###############################################################################  
@app.route('/')
def index():
    return render_template('index.html')

###############################################################################
# Image Classification API
###############################################################################    
@app.route('/imageclassification', methods=['GET','POST','OPTIONS'])
def imageclassification():
    return __initialize_session("image_classification")

@app.route('/imageclassification/imgclassres/iciptimg', methods=['GET'])
def iciptimg():
    return (__output_img("input_image", "image_classification"))
    
@app.route('/imageclassification/imgclassres', methods=['GET'])
def imgclassres():
    return __response_res("image_classification")
    
###############################################################################
# Object Detection API
###############################################################################   
@app.route('/objectdetection', methods=['GET','POST','OPTIONS'])
def objectdetection():
    return __initialize_session("object_detection")
        
@app.route('/objectdetection/odres', methods=['GET'])
def odres():
    return __response_res("object_detection")
        
@app.route('/objectdetection/odres/odiptimg', methods=['GET'])
def odiptimg():
    return (__output_img("input_image", "object_detection"))
          
@app.route('/objectdetection/odres/odresimg', methods=['GET'])
def odresimg():
    return (__output_img("result_image", "object_detection"))        

###############################################################################
# Main Entry
###############################################################################         
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(\
        '--runningmode',\
        type=str,\
        default='runtime',\
        help='running mode: runtime(wait for new task) or job(call to run)'\
    )
    parser.add_argument(\
        '--odpoolpath',\
        type=str,\
        default=os.path.join('.','pooling','object_detection'),\
        help='odject detection pooling path'\
    )
    parser.add_argument(\
        '--icpoolpath',\
        type=str,\
        default=os.path.join('.','pooling','image_classification'),\
        help='image classification pooling path'\
    ) 
    FLAGS, unparsed = parser.parse_known_args()
    
    RUNNING_MODE = FLAGS.runningmode
    OD_POOL_PATH = FLAGS.odpoolpath
    IC_POOL_PATH = FLAGS.icpoolpath
    
    app.run(host='0.0.0.0', debug=False, threaded=True)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    