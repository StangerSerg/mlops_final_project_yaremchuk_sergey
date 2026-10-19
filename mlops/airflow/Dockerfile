FROM apache/airflow:2.10.5-python3.12

# Зависимости кода проекта (src/) поверх образа Airflow.
# apache-airflow фиксируем, чтобы pip не обновил его вместе с зависимостями.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /tmp/requirements.txt

USER root
RUN mkdir -p /opt/airflow/work /opt/airflow/project \
    && chown -R airflow:root /opt/airflow/work /opt/airflow/project
USER airflow
