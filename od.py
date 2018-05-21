# coding: utf-8
# author: jiankaiwang (https://jiankaiwang.no-ip.biz)

import numpy as np
import os
import sys
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import argparse
import configparser
import codecs
import json
import datetime
import time

if tf.__version__ < '1.4.0':
    print('Error: Please upgrade your tensorflow installation to v1.4.* or later!')
    sys.exit(1)

###############################################################################
# Model preparation
############################################################################### 

# Variables
FLAGS = None
config = None
SESS_PATH = None
MODEL_NAME = None
PATH_TO_CKPT = None
PATH_TO_LABELS = None
NUM_CLASSES = 0
SESSNAME = None
#print("Model Name:{}".format(MODEL_NAME))

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
def import_graph():
    global detection_graph
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

# Loading label map
category_index = None
def load_label_map():
    global category_index, PATH_TO_LABELS, NUM_CLASSES
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
#print(categories)

# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

###############################################################################
# Detection
###############################################################################
def __parse_image_size(strconfig):
    strsplit = strconfig.strip().split(',')
    return (int(strsplit[0]), int(strsplit[1]))
    
def __read_session_process():
    global SESS_PATH, FLAGS, SESSNAME
    process_file = os.path.join(SESS_PATH, SESSNAME, 'process.json')
    tmpContent = ""
    with codecs.open(process_file, 'r', 'utf-8') as fin:
        for line in fin:
            tmpContent += line.strip()
    return tmpContent
    
def __get_input_image_name():
    global SESS_PATH, FLAGS
    process = json.loads(__read_session_process())
    return process['input_image']

# keyValueList: [{key="",value=""}]
def __write_session_process(keyValueList):
    global SESS_PATH, FLAGS, SESSNAME
    process_file = os.path.join(SESS_PATH, SESSNAME, 'process.json')
    process = json.loads(__read_session_process())
    for pair in keyValueList:
        process[pair["key"]] = pair["value"]
    with codecs.open(process_file, 'w', 'utf-8') as fout:
        fout.write(json.dumps(process))

def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

def output_object_detection():    
    global config, SESS_PATH, FLAGS, SESSNAME
    
    # test image environment
    task_path = os.path.join(SESS_PATH, SESSNAME)
    image_path = os.path.join(task_path, __get_input_image_name())
    image = Image.open(image_path)
    
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    #print(output_dict)
    
    # Visualization of the results of a detection.
    # visualize_boxes_and_labels_on_image_array(image, boxes, classes, scores, category_index=category_index, **args)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        max_boxes_to_draw=int(config['max_boxes_to_draw']),
        min_score_thresh=float(config['min_score_thresh']),
        line_thickness=int(config['line_thickness']))
    plt.figure(figsize=__parse_image_size(config['IMAGE_SIZE']))
    plt.imshow(image_np)
    plt.axis('off')
    output_res_img = 'output.png'
    plt.savefig(os.path.join(task_path, output_res_img), transparent=True)
    
    # write state
    __write_session_process([\
        {"key":"state","value":"complete"}\
        , {"key":"message","value":"Object detection finished on {}".format(datetime.datetime.now().strftime("%H:%M:%S"))}\
        , {"key":"result_image","value":output_res_img}])

def remove_task_tag():
    global SESSNAME
    targetFile = os.path.join(POOLING_PATH, SESSNAME)
    if os.path.isfile(targetFile):
        os.remove(targetFile)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(\
        '--config',\
        type=str,\
        default=os.path.join('.','usr','object_detection.config'),\
        help='the file to configurate the object detection'\
    )
    parser.add_argument(\
        '--usedconfig',\
        type=str,\
        default='default',\
        help='used model in object detection'\
    )
    parser.add_argument(\
        '--poolingdir',\
        type=str,\
        default=os.path.join('.','pooling','object_detection'),\
        help='pooling directory for object detection'\
    )
    parser.add_argument(\
        '--sessname',\
        type=str,\
        default='',\
        help='assigned session name'\
    )    
    parser.add_argument(\
        '--sesspath',\
        type=str,\
        default=os.path.join('.','session'),\
        help='the file to configurate the object detection'\
    )
    parser.add_argument(\
        '--checktimeperoid',\
        type=int,\
        default=2,\
        help='check pooling directory in seconds'\
    )
    parser.add_argument(\
        '--runningmode',\
        type=str,\
        default='runtime',\
        help='running mode: runtime(wait for new task) or job(call to run)'\
    )
    FLAGS, unparsed = parser.parse_known_args()
    config = configparser.ConfigParser()
    config.read(FLAGS.config)
    config = config[FLAGS.usedconfig]

    # set the parameter
    SESS_PATH = FLAGS.sesspath    
    MODEL_NAME = "local"
    PATH_TO_CKPT = os.path.join('.', config['USR_PATH'], config['PATH_TO_CKPT'])
    PATH_TO_LABELS = os.path.join('.', config['USR_PATH'], config['PATH_TO_LABELS'])
    NUM_CLASSES = int(config['NUM_CLASSES'])
    POOLING_PATH = FLAGS.poolingdir
    CHECK_POOLING_SEC = FLAGS.checktimeperoid
    RUNNING_MODE = FLAGS.runningmode
    SESSNAME = ""

    # start object detection
    import_graph()
    load_label_map()
    
    if RUNNING_MODE == "runtime":
        while True:
            ttlFiles = next(os.walk(POOLING_PATH))[2]       
            if len(ttlFiles) > 0:
                for task in ttlFiles:
                    crtTime = time.ctime()
                    SESSNAME = task
                    try:
                        output_object_detection()
                    except:
                        __write_session_process([\
                            {"key":"state","value":"failure"}\
                            , {"key":"message","value":"Object detection failed on {}"\
                               .format(datetime.datetime.now().strftime("%H:%M:%S"))}])
                    remove_task_tag()
                    endTime = time.ctime()
                    print("State: Parse task {} from {} to {}."\
                          .format(task, crtTime, endTime))
            else:
                time.sleep(CHECK_POOLING_SEC)
    elif RUNNING_MODE == "job":
        SESSNAME = FLAGS.sessname
        output_object_detection()
        

    

    
    
    
    
    
    
    
    
    
    
    
    
    
    