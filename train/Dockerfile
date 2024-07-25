# 베이스 이미지 선택
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# 작업 디렉토리 설정
WORKDIR /opt/ml/code

# File 복사
COPY requirements.txt .
COPY train.py .

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc
 	
# Python 패키지 설치
RUN apt-get update
RUN apt-get install -y git 

RUN pip install --upgrade pip
RUN pip install sagemaker-training
RUN pip install --no-cache-dir -r requirements.txt

# 환경 변수 설정
ENV SAGEMAKER_PROGRAM train.py
