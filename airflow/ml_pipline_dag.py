from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

from src.ingestion.load_data import run_ingestion
from src.validation.validate_data import run_validation
from src.preprocessing.preprocess_data import run_preprocessing
from src.training.run_training import run_training
from src.evaluation.run_evaluation import run_evaluation

default_args = {
    "owner": "mlops",
    "start_date": datetime(2025,1,1)
}

dag = DAG(
    "productivity_ml_pipeline",
    default_args=default_args,
    schedule_interval="@weekly"
)

ingestion = PythonOperator(
    task_id="data_ingestion",
    python_callable=run_ingestion,
    dag=dag
)

validation = PythonOperator(
    task_id="data_validation",
    python_callable=run_validation,
    dag=dag
)

preprocessing = PythonOperator(
    task_id="preprocessing",
    python_callable=run_preprocessing,
    dag=dag
)

training = PythonOperator(
    task_id="training",
    python_callable=run_training,
    dag=dag
)

evaluation = PythonOperator(
    task_id="evaluation",
    python_callable=run_evaluation,
    dag=dag
)

ingestion >> validation >> preprocessing >> training >> evaluation