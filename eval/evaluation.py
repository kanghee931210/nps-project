import argparse

import json
import os
import tarfile

import warnings

import pandas as pd
from autogluon.multimodal import MultiModalPredictor

import datetime
import logging
import boto3

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name',type=str , default=os.environ['model_name'])
    parser.add_argument('--class-type', type=str, default=os.environ['class_type'])
    return parser.parse_args()

def save_evaluation_report(report, output_dir):
    local_path = os.path.join(output_dir, "evaluation.json")
    with open(local_path, "w") as f:
        json.dump(report, f)
    return local_path

def model_fn(model_dir):
    model_path = f"/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path="/opt/ml/processing/model")
    try:
        model = MultiModalPredictor.load(model_dir)
        return model
    except Exception:
        logging.exception("체크포인트에서 모델을 로드하는 데 실패했습니다.")
        logging.exception(f"model_dir = {os.listdir('.')}")
        logging.exception(f"현재 디렉토리 = {os.getcwd()}")
        logging.exception(f"현재 model_dir: {os.listdir(model_dir)}")
        if not os.path.exists(model_dir):
            logging.error(f"오류: {model_dir} 위치에 모델 파일이 존재하지 않습니다.")
        if not os.access(model_dir, os.R_OK):
            logging.error(f"오류: {model_dir} 파일에 읽기 권한이 없습니다.")
        return None

def upload_to_s3(local_file_path, bucket_name, s3_file_path):
    s3 = boto3.client('s3')
    s3.upload_file(local_file_path, bucket_name, s3_file_path)
    print(f'Uploaded {local_file_path} to s3://{bucket_name}/{s3_file_path}')

if __name__ == "__main__":
    args = parse_args()
    current_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    # model load
    model = model_fn("/opt/ml/processing/model/")

    # Data load
    Data = pd.read_csv("/opt/ml/processing/input_data/dataset.csv")

    # predict Data
    prediction = model.predict(Data)

    acc = sum(prediction == Data[args.class_type]) / len(Data)

    report = {
        'Acc' : acc,
        'Folder_name' : args.model_name,
        'Date_Time' : current_time
    }

    output_dir = "/opt/ml/processing/output/evaluation"
    os.makedirs(output_dir, exist_ok=True)
    evaluation_path = os.path.join(output_dir, "evaluation.json")

    with open(evaluation_path, "w") as f:
        json.dump(report, f)

    # local_report_path = save_evaluation_report(report, '/opt/ml/processing/evaluation')

    if not os.path.isfile('/opt/ml/processing/readerboard_in/readerboard.csv'):
        DF = pd.DataFrame([report])
    else:
        DF = pd.read_csv('/opt/ml/processing/readerboard_in/readerboard.csv')
        DF = pd.concat([DF,pd.DataFrame([report])], ignore_index=True)
        DF = DF.sort_values(by='Acc', ascending=False)

    DF.to_csv('/opt/ml/processing/readerboard_out/readerboard.csv', index=False)