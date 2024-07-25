import json
import logging
import os
import pandas as pd
from autogluon.multimodal import MultiModalPredictor
from io import StringIO
import flask
from flask import Flask, request, jsonify, Response
import time

app = Flask(__name__)

# 모델 로드 함수
def model_fn(model_dir, retry_count=3, retry_delay=5):
    for attempt in range(retry_count):
        try:
            model = MultiModalPredictor.load(model_dir)
            logging.info("모델 로드 성공")
            return model
        except Exception as e:
            logging.exception("체크포인트에서 모델을 로드하는 데 실패했습니다.")
            logging.exception(f"model_dir = {os.listdir('.')}")
            logging.exception(f"현재 디렉토리 = {os.getcwd()}")
            logging.exception(f"현재 model_dir: {os.listdir(model_dir)}")
            logging.exception(f"예외: {str(e)}")
            if not os.path.exists(model_dir):
                logging.error(f"오류: {model_dir} 위치에 모델 파일이 존재하지 않습니다.")
            if not os.access(model_dir, os.R_OK):
                logging.error(f"오류: {model_dir} 파일에 읽기 권한이 없습니다.")
            if attempt < retry_count - 1:
                logging.info(f"{retry_delay}초 후에 재시도합니다...")
                time.sleep(retry_delay)
            else:
                return None

model = model_fn('/opt/ml/model', retry_count=3, retry_delay=3)

# 예측 함수
def predict_fn(input_data, model):
    prediction = model.predict(input_data)
    return prediction

# 출력 데이터 처리 함수
def output_fn(prediction, response_content_type, input_df):
    if response_content_type == 'application/json':
        result = prediction.numpy().tolist()
        return json.dumps(result, ensure_ascii=False)
    elif response_content_type == 'text/csv':
        input_df['answer'] = list(prediction)
        return input_df.to_csv(index=False)
    else:
        raise ValueError(f'지원되지 않는 콘텐츠 유형입니다: {response_content_type}')

@app.route('/ping', methods=['GET'])
def ping():
    # Check if the classifier was loaded correctly
    health = model is not None
    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@ app.route('/invocations', methods=['POST'])
def transformation():
    # 입력 데이터 처리

    if 'file' not in request.files and not request.is_json:
        return jsonify({'error': 'No file part or JSON data'}), 400
    if request.files:
        file = request.files['file']
        data = pd.read_csv(file)
    elif request.is_json:
        data = request.get_json()
        data = pd.DataFrame(data) # [] 제거 됨 

    # 예측 수행
    prediction = predict_fn(data, model)

    # # 출력 데이터 처리
    # response_content_type = request.headers.get('Accept', 'application/json')
    # output = output_fn(prediction, response_content_type, [1,2,3,4])
    response = Response(
        response=json.dumps(prediction.to_list(), ensure_ascii=False),
        status=200,
        mimetype='application/json'
    )
    return response