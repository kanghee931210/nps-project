import pandas as pd
import os

import mlflow

from autogluon.multimodal import MultiModalPredictor
from mlflow import log_metric, log_param, log_artifact

if __name__ == '__main__':
    model_dir = os.environ.get('SM_MODEL_DIR','/opt/ml/model')
    train_type = os.environ.get('SM_TRAIN_TYPE','Category2')

    experiment_model = os.environ.get('EX_MODEL','microsoft/deberta-v3-large')
    experiment_loss = os.environ.get('EX_LOSS','focal_loss')
    experiment_epochs = os.environ.get('EX_EPOCH', 5)
    experiment_batch = os.environ.get('EX_BATCH', 1)
    experiment_gpu_batch = os.environ.get('EX_GPU_PER_BATCH', 4)

    train = os.environ.get('SM_CHANNEL_TRAINING','opt/ml/data/training') 
    
    train_path = 'opt/ml/data/training'
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_ARN'])
    mlflow.set_experiment(os.environ['EXPERIMENT_NAME'])
    with mlflow.start_run():
        hyperparameters2 = {
            "model.hf_text.checkpoint_name": experiment_model,
            "optimization.loss_function": experiment_loss,
            "optimization.focal_loss.gamma": 5.0,
            "optimization.focal_loss.reduction": "sum",
            "optimization.max_epochs": experiment_epochs,
            "env.per_gpu_batch_size": experiment_gpu_batch,
            'env.batch_size':experiment_batch
        }

        log_param('hyperparameters', hyperparameters2)

        input_files = [os.path.join(train, file) for file in os.listdir(train) if
                       os.path.isfile(os.path.join(train, file))]
        input_files = [file for file in input_files if file.lower().endswith(".csv")]

        if len(input_files) == 0:
            raise ValueError(('There are no files in {}.\n' +
                              'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                              'the data specification in S3 was incorrectly specified or the role specified\n' +
                              'does not have permission to access the data.').format(train_path, "train"))
        raw_data = [pd.read_csv(file, engine="python") for file in input_files]
        train_data = pd.concat(raw_data)

        predictor = MultiModalPredictor(label=train_type, eval_metric='acc', path=model_dir)
        predictor.fit(hyperparameters=hyperparameters2, train_data=train_data)

        fit_summary = predictor.fit_summary()
        # log_metric('accuracy', fit_summary['val_acc'])  # fit_summary에서 정확도 값을 가져옴
        log_artifact(model_dir)