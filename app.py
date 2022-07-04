from flask import Flask, request, make_response
import tempfile
from pathlib import Path
import os
from subprocess import Popen, PIPE
from yolo_classes import PERSON, VEHICLE, classes_converter
import librosa
from sound.support import calculate_levels, calculate_levels_by_frequencies
import cv2
import io
import requests
import json


SEGMENTATION_URL = f'{os.environ["SEGMENTATION_SERVER"]}/process?segment&rgb'
YOLO_URL = f'{os.environ["YOLO_SERVER"]}/process?yolo'


app = Flask("video-processing-server")

PROCESSING_N = int(os.environ["PROCESSING_N"])
PROCESSING_WIDTH = int(os.environ["PROCESSING_WIDTH"])


@app.route('/')
def health_check():
    return make_response('healthy', 200)


# 오디오 정보만 계산해서 반환
@app.route('/audio_analysis', methods=['POST'])
def audio_analysis():
    audio = request.files.get('audio')
    if audio is None:
        return make_response("there is no given audio file", 400)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td)
        temp_filename = path / 'uploaded_audio'
        audio.save(temp_filename)

        wav, sr = librosa.load(temp_filename, sr=None)

        result = calculate_levels(wav, sr)
        return make_response(result, 200)


# 영상파일을 받아 PROCESS_N 만큼 쪼갠 다음 Yolo, Segmentation 서버에 요청을 보내서 값들을 받아오고, 오디오 정보를 계산해서 반환
@app.route('/process', methods=['POST'])
def process_image():
    video = request.files.get('video')
    if video is None:
        return make_response("there is no given video file", 400)

    segment_results = []
    yolo_results = []
    with tempfile.TemporaryDirectory() as td:
        path = Path(td)
        temp_filename = path / 'uploaded_video'
        resized_video_filename = path / 'resized_video.mp4'
        temp_audio_filename = path / 'audio.wav'
        video.save(temp_filename)

        # ffmpeg process
        command = ["ffmpeg", "-i", temp_filename, "-ac", "1", temp_audio_filename]
        p1 = Popen(command, stdin=PIPE, stdout=PIPE, stderr=PIPE)

        command = ["ffmpeg", "-i", temp_filename, "-vf", f"scale={PROCESSING_WIDTH}:-1", resized_video_filename]
        p2 = Popen(command, stdin=PIPE, stdout=PIPE, stderr=PIPE)

        p1.communicate()
        p2.communicate()

        # audio analysis
        wav, sr = librosa.load(temp_audio_filename, sr=None)
        audio_levels = calculate_levels(wav, sr)

        cap = cv2.VideoCapture(str(temp_filename))
        if cap.isOpened():
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            interval = int(length / (PROCESSING_N + 1))
            for i in range(PROCESSING_N):
                cap.set(cv2.CAP_PROP_POS_FRAMES, (i+1)*interval)
                ret, frame = cap.read()
                if ret:
                    ret, frame = cv2.imencode('.jpg', frame)
                    if ret:
                        frame = frame.tobytes()
                        files = {'image': io.BytesIO(frame)}
                        segment_response = requests.post(SEGMENTATION_URL, files=files)
                        if segment_response.status_code == 200:
                            segment_results.append(json.loads(segment_response.text))
                        else:
                            print(f'request to segmentation server failed: {segment_response.status_code}')
                            continue
                        files = {'image': io.BytesIO(frame)}
                        yolo_response = requests.get(YOLO_URL, files=files)
                        if yolo_response.status_code == 200:
                            yolo_results.append(json.loads(yolo_response.text))
                        else:
                            print(f'request to yolo server failed: {yolo_response.status_code}, {yolo_response.text}')
                            continue
                    else:
                        continue
                else:
                    continue
            cap.release()

    result = {
        'segment': {},
        'rgb_info': {'r': {}, 'g': {}, 'b': {}},
        'yolo': {PERSON: 0.0, VEHICLE: 0.0},
        'audio': audio_levels
    }
    for i in range(len(segment_results)):
        for key, val in segment_results[i]['segment'].items():
            if key in result['segment']:
                result['segment'][key] += val / len(segment_results)
            else:
                result['segment'][key] = val / len(segment_results)
        for col in segment_results[i]['rgb_info'].keys():
            col_info = segment_results[i]['rgb_info'][col]
            for key, val in col_info.items():
                if key in result['rgb_info'][col]:
                    result['rgb_info'][col][key] += val / len(segment_results)
                else:
                    result['rgb_info'][col][key] = val / len(segment_results)
    for yolo_result in yolo_results:
        for key, val in yolo_result.items():
            category = classes_converter[key]
            if category is not None:
                result['yolo'][category] += val / len(yolo_results)

    return make_response(result, 200)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
