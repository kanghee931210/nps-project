# 베이스 이미지 선택
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# 작업 디렉토리 설정
WORKDIR /opt/ml/processing

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python", "evaluation.py"]