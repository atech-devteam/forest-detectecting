# -*- coding: UTF-8 -*-
"""
Flask Micro WAS for detecting objects from a video of the mountain area taken by a drone
Using Mark RCNN in mrcnn module and detector module
Upload File Directory : ./static/data/{session-id}
- session-id implemented by uuid
- detected result will be stored in ./data/{session-id}/(low-fps.mp4, high-fps.mp4)/detected_obj.json 
Made by JYLee, Sam Lee 
Last updated on Oct, 2020 
"""
import logging 
import configparser
import os, base64
from flask import Response, Flask, render_template, request, make_response, session, json, send_file, copy_current_request_context, stream_with_context, send_from_directory
from flask_session import Session
from detector.markrcnn_module import Detector
import re 
import time
from threading import Thread
import cv2
import skvideo.io
import skvideo.datasets
import imageio
import base64
import threading
import numpy as np
from shapely.geometry import Polygon
from skimage.measure import find_contours
import json

# declation constants
STATUS_HIGH = 'high_state'
STATUS_LOW = 'low_state'
UPDATE_MODE = 'update'
READ_MODE = 'read'
PROCESS_STATE_IDLE = 'idle'
PROCESS_STATE_ONGOING = 'ongoing'
PROCESS_STATE_DONE = 'finished'
RESULT_FILE_NAME = 'result.mp4'
IMAGES_DIR = 'images'
HIGH_IMAGE_DIR = 'high_fps_images'
LOW_JSON_DIR = 'low_json'
HIGH_JSON_DIR = 'high_json'
CHUNK_SIZE = 8192

# Read config.ini
config = configparser.ConfigParser()
ROOT_PATH = os.path.abspath('./')
config.read(os.path.join(ROOT_PATH, 'config.ini'))
session_cache_dir = os.path.join(ROOT_PATH, 'cache', 'session') 
data_dir = os.path.join(ROOT_PATH, config['SERVICE']['DATA_DIR'])

# create directoris for session and data if the directories don't exist.
os.makedirs(session_cache_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# define persons's classes from config.ini
PERSON_CLASS_IDS = [ int(str) for str in config['MODEL']['PERSON_CLASS_IDS'].split(',') ]

# to use logging
log = logging.getLogger('forestry_detection')

# Flask App 
app = Flask(__name__)
# Set the secret key to some random bytes. Keep this really secret!
# app.secret_key = b'input_your_secret_key'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = session_cache_dir
app.secret_key = os.urandom(32).hex()
app.config['TEMPLATES_AUTO_RELOAD'] = True #  settup for auto reload when static files are changed
Session(app)

# Detecting using marskrncc
detector = Detector(ROOT_PATH)

def generate_session():
    return str(time.time())


@app.route('/')
def index():
    if 'sid' not in session :
        session['sid'] = generate_session()
        session['file_name'] = ''
        session['upload_state'] = False
        update_process_status(session['sid'], PROCESS_STATE_IDLE, STATUS_LOW)
        update_process_status(session['sid'], PROCESS_STATE_IDLE, STATUS_HIGH)
        session[STATUS_HIGH] = PROCESS_STATE_IDLE
        session[STATUS_LOW] = PROCESS_STATE_IDLE
        session['num_low_frames'] = 0

    sid = session['sid']
    
    return render_template('index.html', app_info={
            'application_name':'forest detecting',
            'sid':sid
    })

@app.route('/clearSession', methods=['GET'])
def clearSession():
    if 'sid' in session:
        sid = session['sid']
    session.clear()
    return make_response(("cleared the session", 200))


@app.route('/upload', methods=['POST'])
def upload():
    # start routine
    if 'upload_state' in session and session['upload_state']==True:
        return make_response(("Already uploaded a video. Clear session to restart", 400))
    file = request.files['file']
    os.makedirs(os.path.join(data_dir, session.get('sid')), exist_ok=True)
    save_path = os.path.join(data_dir, session.get('sid'), file.filename) 
    current_chunk = int(request.form['dzchunkindex'])

    # If the file already exists it's ok if we are appending to it,
    # but not if it's new file that would overwrite the existing one
    if os.path.exists(save_path) and current_chunk == 0:
        # 400 and 500s will tell dropzone that an error occurred and show an error
        return make_response(('File already exists', 400))
    
    try:
        with open(save_path, 'ab') as f:
            f.seek(int(request.form['dzchunkbyteoffset']))
            f.write(file.stream.read())
    except OSError:
        # log.exception will include the traceback so we can see what's wrong
        log.exception('Could not write to file')
        return make_response(("Not sure why,"
                              " but we couldn't write the file to disk", 500))

    total_chunks = int(request.form['dztotalchunkcount'])

    if current_chunk + 1 == total_chunks:
        # This was the last chunk, the file should be complete and the size we expect
        if os.path.getsize(save_path) != int(request.form['dztotalfilesize']):
            log.error(f"File {file.filename} was completed, "
                      f"but has a size mismatch."
                      f"Was {os.path.getsize(save_path)} but we"
                      f" expected {request.form['dztotalfilesize']} ")
            return make_response(('Size mismatch', 500))
        else:
            log.info(f'File {file.filename} has been uploaded successfully')
            session['upload_state'] = True
            session['file_name'] = file.filename
            
            # thread low fps start
            t_low = threading.Thread(target=gen_detected_imgs_from_video, args=(session['sid'], session['file_name'], 'LOW' )).start()
            
            # thread high fps start
            # t_hight = threading.Thread(target=gen_detected_video_from_video, args=(session['sid'], session['file_name'] )).start()
            t_high = threading.Thread(target=gen_detected_imgs_from_video, args=(session['sid'], session['file_name'], 'HIGH' )).start()
            
    else:
        log.debug(f'Chunk {current_chunk + 1} of {total_chunks} '
                  f'for file {file.filename} complete')
    
    return make_response(("Chunk upload successful", 200))

@app.route('/getState', methods=['GET'])
def getState():
    sid = session['sid']
    file_name = session['file_name']
    static_data_dir = config['SERVICE']['DATA_DIR']
    
    img_dir = os.path.join(data_dir, session['sid'], IMAGES_DIR)
    # print(img_dir)
    current_ongoing_detected_file = None
    result_json = {}
    try:
        file_list_images = [file for file in os.listdir(img_dir) if file.endswith(".jpg")]
        # print(file_list_images)
        low_frame_num = len(file_list_images)
        if low_frame_num > 0:
            # sorted_file_list = sorted(file_list_images, reverse=True)
            file_list_images.sort(key=lambda f: int(re.sub('\D', '', f)), reverse=True)
            current_ongoing_detected_file = f'/{static_data_dir}/{sid}/{IMAGES_DIR}/{file_list_images[0]}'
            rJsonDir = os.path.join(data_dir, sid, LOW_JSON_DIR)
            frame_num = os.path.splitext(file_list_images[0])[0]
            with open(os.path.join(rJsonDir, f'{frame_num}.json'), 'r') as f:
                result_json = json.load(f)
    except:
        low_frame_num = 0
    high_state = read_process_status(sid, STATUS_HIGH)
    low_state = read_process_status(sid, STATUS_LOW)


    # try:
    #     low_frame_num = session['low_frame_num']
    # except:
    #     low_frame_num = 0
    
    return app.response_class(
            response=json.dumps({
                'upload_state':session['upload_state'],
                'detecting_high_state':high_state,
                'detecting_low_state':low_state,
                'file_name':session['file_name'],
                'low_frame_num' : low_frame_num,
                'current_detected_file' : current_ongoing_detected_file,
                'result' : result_json
            }),
            status=200,
            mimetype='application/json'
        )

@app.route('/getImage/<int:frame_num>')
def getDetectedImg(frame_num):
    sid = session['sid']
    static_data_dir = config['SERVICE']['DATA_DIR']
    rJsonDir = os.path.join(data_dir, sid, LOW_JSON_DIR)
    with open(os.path.join(rJsonDir, f'{frame_num}.json'), 'r') as f:
        result_json = json.load(f)

    return app.response_class(
            
            response=json.dumps({
                'src':f'/{static_data_dir}/{sid}/{IMAGES_DIR}/{frame_num}.jpg',
                'result':result_json
            }),
            status=200,
            mimetype='application/json'
        )

@app.route('/result_file_download')
def result_file_download():
    sid = session['sid']
    resultFile = os.path.join(data_dir, session.get('sid'), 'dvideo', RESULT_FILE_NAME)
    if not os.path.exists(resultFile):
        resultDir = os.path.join(data_dir, sid, 'dvideo')
        os.makedirs(resultDir, exist_ok=True)

        inputFileName = os.path.join(data_dir, session['sid'], session['file_name'])
        fps, width, height = get_input_video_info(inputFileName)

        resize_width = int(config['MODEL']['RESULT_FRAME_WIDTH'])
        new_size = ( resize_width, int(height*resize_width / width))
        new_fps =  int(config['MODEL']['HIGH_SAMPLING_RATE'])

        v_out = cv2.VideoWriter(
                os.path.join(resultDir, RESULT_FILE_NAME), 
                cv2.VideoWriter_fourcc(*'mp4v'), 
                new_fps,  
                new_size )
        
        result_images_dir = os.path.join(data_dir, sid, HIGH_IMAGE_DIR)
        file_names = next(os.walk(result_images_dir))[2]
        file_names.sort(key=lambda f: int(re.sub('\D', '', f)), reverse=True)
        
        for file in sorted(file_names):
            if file.find('.jpg'):
                img = cv2.imread(os.path.join(result_images_dir, file))
                v_out.write(img)
        v_out.release()

    if os.path.exists(resultFile):
        return Response(
            stream_with_context(read_file_chunks(resultFile)),
            headers={
                'Content-Disposition': f'attachment; filename={RESULT_FILE_NAME}'
            }
        )
    else:
        raise exc.NotFound()

def read_file_chunks(path):
    with open(path, 'rb') as fd:
        while 1:
            buf = fd.read(CHUNK_SIZE)
            if buf:
                yield buf
            else:
                break
    
def update_process_status(sid, value=PROCESS_STATE_IDLE, status_type=STATUS_LOW):
    os.makedirs(os.path.join(data_dir, sid), exist_ok=True)
    doc_filename = os.path.join(data_dir, sid, f'{status_type}.txt')
    
    with open(doc_filename, 'w') as file:
        file.write(value)

def read_process_status(sid, status_type=STATUS_LOW):
    doc_filename = os.path.join(data_dir, sid, f'{status_type}.txt')
    
    with open(doc_filename, 'r') as file:
        status = file.readline()
    return status        

#thread method
def gen_detected_imgs_from_video(sid, file_name, mode='LOW'):
    # sid = session['sid']

    if mode == 'LOW':
        imgDir = os.path.join(data_dir, sid, IMAGES_DIR)
    else:
        imgDir = os.path.join(data_dir, sid, HIGH_IMAGE_DIR)
    inputFileName = os.path.join(data_dir, sid, file_name)

    os.makedirs(imgDir, exist_ok=True)
    vid = skvideo.io.FFmpegReader(inputFileName)
    # data = skvideo.io.ffprobe(inputFileName)['video']
    # v_rate = data['@r_frame_rate']
    # num_frames, v_x, v_y, channels = vid.getShape()
    fps, width, height = get_input_video_info(inputFileName)
    if mode == 'LOW':
        sampling_rate_4extraction = round(fps / int(config['MODEL']['LOW_SAMPLING_RATE']))
        update_process_status(sid, PROCESS_STATE_ONGOING, STATUS_LOW)
    else:
        sampling_rate_4extraction = round(fps / int(config['MODEL']['HIGH_SAMPLING_RATE']))
        update_process_status(sid, PROCESS_STATE_ONGOING, STATUS_HIGH)
    
    workCount = 0
    # print(f'mode:{mode}, sampling_rate : {sampling_rate_4extraction}')
    
    for idx, frame in enumerate(vid.nextFrame()):
        if idx % sampling_rate_4extraction == 0:
            # print(f'{mode} - working frame : {workCount}')
            r, classNames, jpeg_frame =  detector.detectionFromMem2Mem(frame, int(config['MODEL']['RESULT_FRAME_WIDTH']))
            cv2.imwrite(os.path.join(imgDir,f'{workCount}.jpg'), jpeg_frame)
            checkOverlaps(r, classNames, sid,  mode, workCount)
            workCount+=1
    vid.close()
    update_process_status(sid, PROCESS_STATE_DONE, STATUS_LOW if mode == 'LOW' else STATUS_HIGH)

"""
a parameter :  a input video file
return : fps, width, height
"""
def get_input_video_info(inputFileName):
    vcap = cv2.VideoCapture(inputFileName)
    width = vcap.get(cv2.CAP_PROP_FRAME_WIDTH )
    height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT )
    fps =  vcap.get(cv2.CAP_PROP_FPS)
    # print('inputfile info', fps, width, height)
    vcap.release()
    return fps, int(width), int(height) 

def checkOverlaps(r, classNames, sid, mode, frame_idx):
    if mode == 'LOW':
        rJsonDir = os.path.join(data_dir, sid, LOW_JSON_DIR)
    else:
        rJsonDir = os.path.join(data_dir, sid, HIGH_JSON_DIR)
    os.makedirs(rJsonDir, exist_ok=True)
    resultRoot = {'person_count':0, 'detect_result':[]}
    personCount = 0
    detect_result = []
    for idx, cid in enumerate(r['class_ids']):
        if cid in PERSON_CLASS_IDS:
            personCount += 1
            person = r['masks'][:,:,idx]
            person_contours = find_contours(person, 0.5)
            isIntersect = False
            personResultItem = {'className':classNames[cid], 'isIntersect':False, 'intersectedClassName':None}
            for jdx, cid_sub in enumerate(r['class_ids']):
                if idx == jdx or cid_sub in PERSON_CLASS_IDS:
                    continue
                mask = r['masks'][:,:,jdx]
                contours_sub = find_contours(mask, 0.5)
                intersectedClassName = None
                for p_c in person_contours:
                    for s_c in contours_sub:
                        p_ploy = Polygon(p_c)
                        s_ploy = Polygon(s_c)
                        if p_ploy.overlaps(s_ploy) or p_ploy.intersects(s_ploy):
                            isIntersect = True
                            intersectedClassName = classNames[cid_sub]
                            break
                    if isIntersect:
                        break
                if isIntersect:
                    break
            personResultItem['isIntersect'] = isIntersect
            personResultItem['intersectedClassName'] = intersectedClassName
            resultRoot['detect_result'].append(personResultItem)
    resultRoot['person_count'] = personCount
    with open(os.path.join(rJsonDir, f'{frame_idx}.json'), "w") as json_file:
        json.dump(resultRoot, json_file)
                        


if __name__ == '__main__':
    app.run(host=config['SERVICE']['HOST_ADDR'],port=int(config['SERVICE']['PORT_NUM']))