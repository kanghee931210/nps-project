# 베이스 이미지 선택
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

COPY NPS_infer /opt/program
COPY requirements.txt  /opt/program
WORKDIR /opt/program


RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python3 \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && \
    pip install flask gevent gunicorn && \
    pip install autogluon==1.1.0 && \
    rm -rf /root/.cache

RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

EXPOSE 8080

ENTRYPOINT ["python"]

# serve is a python script under code/ directory that launches nginx and gunicorn processes
CMD ["serve"]