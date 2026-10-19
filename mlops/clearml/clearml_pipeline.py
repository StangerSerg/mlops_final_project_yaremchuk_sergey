"""ClearML-пайплайн: обучение CatBoost на разных гиперпараметрах и выбор лучшей.

Шаги:
    train_candidate (xN) — по одному на кандидата из search_space.json:
        обучение на train с early stopping по valid, метрики на valid.
    select_best_model     — кандидат с максимальным PR-AUC на valid,
        оценка на отложенном test, загрузка ranker.pkl в S3
        (S3_PREFIX/models или MODEL_S3_KEY) и регистрация в Model Registry.

Данные — обучающая выборка из S3_PREFIX/processed_data (результат
Airflow-DAG preprocessing_data). Параметры окружения (ClearML, S3)
читаются из .env.

Запуск из корня репозитория:
    PYTHONPATH=. python mlops/clearml/clearml_pipeline.py --mode local
    PYTHONPATH=. python mlops/clearml/clearml_pipeline.py --mode remote --queue default
"""

import argparse
import os
from pathlib import Path
from typing import Any

from clearml import PipelineDecorator
from dotenv import load_dotenv

# до декораторов: имя проекта и настройки ClearML берутся из окружения
load_dotenv()

PROJECT = os.environ.get("CLEARML_PROJECT", "mlops")
PIPELINE_NAME = "Train catboost model"
DEFAULT_SEARCH_SPACE = "mlops/clearml/search_space.json"
# Для запуска шагов на clearml-agent: откуда агент клонирует код (src/)
REPO = os.environ.get("CLEARML_PIPELINE_REPO") or None
STEP_PACKAGES = [
    "boto3~=1.40.69",
    "catboost~=1.2.8",
    "joblib~=1.5.2",
    "numpy~=2.2.5",
    "pandas~=2.2.3",
    "pandera==0.24.0",
    "pyarrow~=22.0.0",
    "pydantic==2.13.5",
    "pydantic-settings>=2.10",
    "scikit-learn~=1.6.1",
]


@PipelineDecorator.component(
    name="train_candidate",
    return_values=["metrics"],
    task_type="training",
    packages=STEP_PACKAGES,
    repo=REPO,
)
def train_candidate(candidate_id: int, params: dict) -> dict:
    from clearml import Task

    from src.models.evaluation import evaluate_model
    from src.models.selection import load_training_frame, split_train_valid_test
    from src.models.training import train_model
    from src.schemas.config import ModelParams
    from src.utils.config import load_config
    from src.utils.settings import get_settings
    from src.utils.storage import ArtifactStorage

    settings = get_settings()
    config = load_config(settings.config_path)
    model_params = ModelParams.model_validate(params)

    x, y = load_training_frame(ArtifactStorage(settings), config)
    x_train, x_valid, _, y_train, y_valid, _ = split_train_valid_test(
        x, y, config.split_params
    )
    model = train_model(x_train, y_train, model_params, eval_set=(x_valid, y_valid))

    metrics = evaluate_model(model, x_valid, y_valid)
    metrics["best_iteration"] = float(model.get_best_iteration() or 0)

    task: Task | None = Task.current_task()
    if task is not None:
        task.connect(model_params.model_dump(), name="model_params")
        task.add_tags([f"candidate-{candidate_id}"])
        logger = task.get_logger()
        for name, value in metrics.items():
            logger.report_single_value(f"valid/{name}", value)
    print(f"candidate {candidate_id}: {params} -> {metrics}")
    return metrics


@PipelineDecorator.component(
    name="select_best_model",
    return_values=["model_uri"],
    task_type="training",
    packages=STEP_PACKAGES,
    repo=REPO,
)
def select_best_model(candidates: list, results: list) -> str:
    import joblib
    import pandas as pd
    from clearml import OutputModel, Task

    from src.models.evaluation import evaluate_model
    from src.models.selection import (
        SELECTION_METRIC,
        load_training_frame,
        select_best,
        split_train_valid_test,
    )
    from src.models.training import train_model
    from src.schemas.config import ModelParams
    from src.utils.config import load_config
    from src.utils.settings import get_settings
    from src.utils.storage import ArtifactStorage

    settings = get_settings()
    config = load_config(settings.config_path)
    storage = ArtifactStorage(settings)

    best = select_best(results)
    best_params = ModelParams.model_validate(candidates[best])
    print(f"best candidate #{best}: {candidates[best]} -> {results[best]}")

    # Модель лучшего кандидата воспроизводится детерминированно
    # (тот же сплит и random_seed) и оценивается на отложенном test.
    x, y = load_training_frame(storage, config)
    x_train, x_valid, x_test, y_train, y_valid, y_test = split_train_valid_test(
        x, y, config.split_params
    )
    model = train_model(x_train, y_train, best_params, eval_set=(x_valid, y_valid))
    test_metrics = evaluate_model(model, x_test, y_test)
    print(f"test metrics: {test_metrics}")

    model_path = storage.local_path("models", config.model_file)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    model_uri = storage.publish("models", config.model_file)

    task: Task | None = Task.current_task()
    if task is not None:
        logger = task.get_logger()
        for name, value in test_metrics.items():
            logger.report_single_value(f"test/{name}", value)
        table = pd.DataFrame(candidates).assign(
            **{f"valid_{k}": [r[k] for r in results] for k in results[0]}
        )
        logger.report_table("candidates", f"valid {SELECTION_METRIC}", table_plot=table)
        task.connect(best_params.model_dump(), name="best_model_params")

    # Model Registry требует ClearML-сервер (в offline-режиме недоступен)
    if task is not None and not Task.is_offline():
        output_model = OutputModel(
            task=task,
            name="ranker",
            framework="CatBoost",
            tags=["best"],
            comment=f"test {SELECTION_METRIC}={test_metrics[SELECTION_METRIC]:.5f}",
        )
        output_model.update_weights(
            weights_filename=str(model_path), auto_delete_file=False
        )
        output_model.set_metadata("s3_uri", model_uri, v_type="str")
    return model_uri


@PipelineDecorator.pipeline(
    name=PIPELINE_NAME,
    project=PROJECT,
    version="1.0.0",
    repo=REPO,
)
def training_pipeline(search_space_path: str = DEFAULT_SEARCH_SPACE) -> Any:
    from src.models.selection import load_search_space
    from src.utils.config import load_config
    from src.utils.settings import get_settings

    config = load_config(get_settings().config_path)
    candidates = [
        c.model_dump()
        for c in load_search_space(Path(search_space_path), config.model_params)
    ]

    # шаги кандидатов запускаются параллельно
    pending = [train_candidate(i, params) for i, params in enumerate(candidates)]
    # обращение к результатам дожидается завершения шагов
    results = [{k: float(v) for k, v in r.items()} for r in pending]

    model_uri = select_best_model(candidates, results)
    print(f"Лучшая модель: {model_uri}")
    return model_uri


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["local", "remote", "debug"],
        default="local",
        help="local — шаги подпроцессами на этой машине; "
        "remote — шаги в очереди clearml-agent; "
        "debug — шаги функциями в одном процессе",
    )
    parser.add_argument("--queue", default="default", help="очередь для --mode remote")
    parser.add_argument("--search-space", default=DEFAULT_SEARCH_SPACE)
    args = parser.parse_args()

    if args.mode == "debug":
        PipelineDecorator.debug_pipeline()
    elif args.mode == "local":
        PipelineDecorator.run_locally()
    else:
        PipelineDecorator.set_default_execution_queue(args.queue)

    training_pipeline(search_space_path=args.search_space)


if __name__ == "__main__":
    main()
