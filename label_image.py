# ==============================================================================
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Copyright 2018 Jiankai Wang. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# ChangeLog: 
# |- import necessary library(configparser, json, os, codecs, datetime, time)
# |- add file input to fetch other settings (beginning from line 99)
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import configparser
import json
import os
import codecs
import datetime
import time
import numpy as np
import tensorflow as tf


def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


def __read_session_process(sessPath, sessName):
    processFile = os.path.join(sessPath, sessName, 'process.json')
    tmpContent = ''
    with codecs.open(processFile, 'r', 'utf-8') as fin:
        for line in fin:
            tmpContent = tmpContent + line.strip()
    return json.loads(tmpContent)

# keyValueList: [{key="",value=""}]    
def __write_session_process(sessPath, sessName, keyValueList): 
    process = __read_session_process(sessPath, sessName)
    process_file = os.path.join(sessPath, sessName, 'process.json')   
    for pair in keyValueList:
        process[pair["key"]] = pair["value"]
    with codecs.open(process_file, 'w', 'utf-8') as fout:
        fout.write(json.dumps(process))

# write out the result
def __write_result(sesspath, sessname, resultData):
  __write_session_process(sesspath, sessname, [\
      {"key":"state","value":"complete"}\
      , {"key":"message","value":"Image classification finished on {}".\
         format(datetime.datetime.now().strftime("%H:%M:%S"))} \
      , {"key":"result","value":json.dumps(resultData)}])        
        
def classify_single_image(sesspath, sessname):
  global input_height, input_width, input_mean, input_std, args
  global graph, input_operation, output_operation, label_file
  
  # read the prcoess file
  preloader = __read_session_process(sesspath, sessname)
      
  file_name = args.image if args.image else \
    os.path.join(sesspath, sessname, preloader['input_image'])
    
  t = read_tensor_from_image_file(
      file_name,
      input_height=input_height,
      input_width=input_width,
      input_mean=input_mean,
      input_std=input_std)

  with tf.Session(graph=graph) as sess:
    results = sess.run(output_operation.outputs[0], {
        input_operation.outputs[0]: t
    })
  results = np.squeeze(results)

  top_k = results.argsort()[-5:][::-1]
  labels = load_labels(label_file)
  resultData = {}
  for i in top_k:
    #print(labels[i], results[i])
    resultData[labels[i]] = str(round(results[i]* 100, 1)) + '%'
  
  # write state
  __write_result(sesspath, sessname, resultData)
  
def remove_task_tag(sessname):
  global pooling_dir
  targetFile = os.path.join(pooling_dir, sessname)
  if os.path.isfile(targetFile):
    os.remove(targetFile)
  
if __name__ == "__main__":
  file_name = "tensorflow/examples/label_image/data/grace_hopper.jpg"
  model_file = \
    "tensorflow/examples/label_image/data/inception_v3_2016_08_28_frozen.pb"
  label_file = "tensorflow/examples/label_image/data/imagenet_slim_labels.txt"
  input_height = 299
  input_width = 299
  input_mean = 0
  input_std = 255
  input_layer = "input"
  output_layer = "InceptionV3/Predictions/Reshape_1"
  
  parser = argparse.ArgumentParser()
  parser.add_argument("--image", help="image to be processed")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  
  # add sesspath, sessname, usedconfig
  parser.add_argument("--sesspath" \
                      , default=os.path.join('.','session') \
                      , help="assign session path")
  parser.add_argument("--sessname" \
                      , default="" \
                      , help="assign session (or task) name")
  parser.add_argument("--config" \
                      , default=os.path.join('.','usr','image_classification.config') \
                      , help="config path")  
  parser.add_argument("--usedconfig" \
                      , default='default' \
                      , help="used config") 
  parser.add_argument('--poolingdir',\
                      type=str,\
                      default=os.path.join('.','pooling','image_classification'),\
                      help='pooling directory for image classification')   
  parser.add_argument('--checktimeperoid',\
                      type=int,\
                      default=2,\
                      help='check pooling directory in seconds')
  parser.add_argument('--runningmode',\
                      type=str,\
                      default='runtime',\
                      help='running mode: runtime(wait for new task) or job(call to run)')  
  args = parser.parse_args()
  
  # get args value
  sesspath = args.sesspath
  sessname = args.sessname
  config_file = args.config
  usedconfig = args.usedconfig
  pooling_dir = args.poolingdir
  check_time_peroid = args.checktimeperoid
  running_mode = args.runningmode

  # add config parser
  config = configparser.ConfigParser()
  config.read(config_file)
  config = config[usedconfig]      

  #############################################################################
  # load first in runtime mode
  #############################################################################
  # from config
  model_file = args.graph if args.graph else \
      os.path.join('.', 'usr', config['GRAPH'])
  graph = load_graph(model_file)    

  # from config
  label_file = args.labels if args.labels else \
      os.path.join('.', 'usr', config['LABELS'])  
      
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std      
    
  # from config
  input_layer = args.input_layer if args.input_layer else config['INPUT_LAYER']
  output_layer = args.output_layer if args.output_layer else config['OUTPUT_LAYER']    

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name)
  output_operation = graph.get_operation_by_name(output_name)
  
  #############################################################################
  # run task in runtime mode
  #############################################################################
  if running_mode == "job":
    classify_single_image(sesspath, sessname)   
  else:      
    while True:
      ttlFiles = next(os.walk(pooling_dir))[2]       
      if len(ttlFiles) > 0:
        for task in ttlFiles:
          crtTime = time.ctime()
          classify_single_image(sesspath, task)
          remove_task_tag(task)
          endTime = time.ctime()
          print("State: Parse task {} from {} to {}."\
                .format(task, crtTime, endTime))
      else:
        time.sleep(check_time_peroid)
