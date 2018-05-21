# coding: utf-8
# authorL jiankaiwang (http://jiankaiwang.no-ip.biz/)

import os
import argparse
import time
import shutil

def cleanSessionData(dirName, crtTime):
    global SESSION_PATH, KEEP_TIME_PEROID
    targetDir = os.path.join(SESSION_PATH, dirName)
    targetDir_mt = os.path.getmtime(targetDir)
    if crtTime - targetDir_mt > KEEP_TIME_PEROID:
        try:
            shutil.rmtree(targetDir, ignore_errors=True)
            print("Warning: Delete session {}.".format(dirName))
            return 0, ""
        except Exception as e:
            return 1, e
    return 0, ""
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(\
        '--sessionpath',\
        type=str,\
        default=os.path.join('.','session'),\
        help='session path'\
    )
    parser.add_argument('--keeptimeperoid',\
        type=int,\
        default=2 * 24 * 60 * 60,\
        help='files to keep in seconds')
    FLAGS, unparsed = parser.parse_known_args()
    
    SESSION_PATH = FLAGS.sessionpath
    KEEP_TIME_PEROID = int(FLAGS.keeptimeperoid)
    
    totalDirName = next(os.walk(SESSION_PATH))[1]
    crtTime = time.time()
    for dirName in totalDirName:
        state, msg = cleanSessionData(dirName, crtTime)
        if state != 0:
            print("Error: Fail to delete file {} and error is {}."\
                  .format(dirName, msg))
        
        