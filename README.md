# Retail Recommendation System

Сервис для e-commerce, который оценивает вероятность покупки товара после просмотра.
Модель CatBoost обучена на логах просмотров/покупок и признаках товаров
(категория, наличие, CTR) с контекстом запроса (час, день недели).
Инференс — FastAPI.

| Сценарий | Что делает сервис |
|---|---|
| Шанс покупки | `GET /predict_purchase` — вероятность покупки товара в заданный час и день недели |

Финальный проект курса MLOps (Яндекс Практикум).

---

## Как устроено

```
S3 raw_data ──► scripts/prepare_data.py ──► S3 processed_data ──► scripts/train_model.py ──► S3 models/ranker.pkl
 (events.csv,     предобработка + фичи        (item_features,        обучение + метрики                │
  item_props)                                  data_for_training)                                      ▼
                                                      └──────── item_features.parquet ──────► FastAPI (src/main.py)
```

* **Хранилище артефактов** (`src/utils/storage.py`). Каждая «область» хранится
  в S3 по пути `S3_PREFIX/<raw_data|processed_data|models>/<файл>` и одновременно
  в локальном каталоге (`data/raw`, `data/prepared`, `models`). Локальный каталог
  служит кэшем для скачанных файлов и **fallback-ом**: если S3 не настроен или
  недоступен, данные и модель берутся из локального каталога.
* **Настройки окружения** (`src/utils/settings.py`) читаются из `.env` или
  переменных окружения: бакет, ключи, префикс, каталоги, порт. Для разных
  окружений (тестовая ВМ / сдача) меняется только `.env`, код остаётся прежним.
* **ML-параметры** (`config/config.json`): имена файлов, признаки, окно таргета,
  параметры разбиения и модели.
* **Сервис** при старте загружает модель и признаки товаров. Если загрузить
  их не удалось, сервис всё равно поднимается, но `/health` и
  `/predict_purchase` отвечают `503`.

---

## Окружение

Установка uv: https://docs.astral.sh/uv/getting-started/installation/

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements-dev.txt   # сервис + ruff/mypy/pytest
# только runtime-зависимости (как в Docker-образе):
# uv pip install -r requirements.txt
```

Альтернатива — `uv sync` по `pyproject.toml` / `uv.lock`.

### Настройки (.env)

```bash
cp .env.example .env
```

| Переменная | Назначение | По умолчанию |
|---|---|---|
| `S3_ENDPOINT_URL`, `S3_REGION` | Object Storage | `https://storage.yandexcloud.net`, `ru-central1` |
| `S3_BUCKET` | бакет; если пусто, S3 не используется | — |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | ключи статического доступа | — |
| `S3_PREFIX` | корень проекта в бакете | `yaremchuk_sergey_project` |
| `S3_RAW_DIR`, `S3_PROCESSED_DIR`, `S3_MODELS_DIR` | подкаталоги | `raw_data`, `processed_data`, `models` |
| `MODEL_S3_KEY` | полный ключ модели, если она лежит не в `S3_PREFIX/models` | — |
| `LOCAL_RAW_DIR`, `LOCAL_PROCESSED_DIR`, `LOCAL_MODELS_DIR` | локальный кэш / fallback | `data/raw`, `data/prepared`, `models` |
| `APP_HOST`, `APP_PORT` | адрес сервиса | `0.0.0.0`, `32000` (в Docker `8000`) |

`.env` не коммитится. В CI эти же имена задаются через GitLab CI/CD Variables
(ключи помечены masked).

---

## Запуск

Сырые данные (`events.csv`, `item_properties_part1.csv`) лежат в S3
`S3_PREFIX/raw_data/` или локально в `data/raw/`. В репозиторий они не
коммитятся.

```bash
# 1. Подготовка признаков -> processed_data (item_features.parquet, data_for_training.parquet)
python -m scripts.prepare_data
python -m scripts.prepare_data --sample 500000   # быстрый прогон на части строк

# 2. Обучение модели -> models/ranker.pkl (+ метрики в лог, в т.ч. PR-AUC)
python -m scripts.train_model

# 3. Сервис
python -m src.main             # http://127.0.0.1:32000, Swagger: /docs
```

### Примеры запросов

```bash
curl "http://127.0.0.1:32000/predict_purchase?itemid=356475&hour=14&weekday=3"
# {"itemid":356475,"purchase_probability":0.0123}

curl http://127.0.0.1:32000/health    # 200 {"status":"ok",...} | 503 если модель не загружена
curl http://127.0.0.1:32000/version
```

| Код | Когда |
|---|---|
| 200 | успешное предсказание |
| 404 | товара нет в feature store |
| 422 | невалидные параметры (`hour` 0–23, `weekday` 0–6, `itemid` ≥ 0) |
| 503 | модель не загружена |

### V2 Inference Protocol (KServe)

Для Kubernetes сервис реализует Open Inference Protocol. Имя модели задаётся
`MODEL_NAME`, по умолчанию `ranker`.

| Запрос | Назначение |
|---|---|
| `GET /v2/health/live` | liveness: процесс отвечает |
| `GET /v2/health/ready` | readiness: модель загружена (иначе 503) |
| `GET /v2/models/ranker` | метаданные: входы `itemid`/`hour`/`weekday` (INT64), выход `purchase_probability` (FP64) |
| `POST /v2/models/ranker/infer` | батч-инференс в формате V2 |
| `GET /v2/models/ranker/infer?itemid=..&hour=..&weekday=..` | упрощённый инференс одной записи |

```bash
curl -X POST http://127.0.0.1:32000/v2/models/ranker/infer -H 'Content-Type: application/json' -d '{
  "inputs": [
    {"name": "itemid",  "shape": [2], "datatype": "INT64", "data": [356475, 98113]},
    {"name": "hour",    "shape": [2], "datatype": "INT64", "data": [14, 3]},
    {"name": "weekday", "shape": [2], "datatype": "INT64", "data": [3, 6]}
  ]}'
# {"model_name":"ranker","model_version":"1","outputs":[{"name":"purchase_probability","shape":[2],"datatype":"FP64","data":[...]}]}
```

Ошибки возвращаются в формате V2: `{"error": "..."}`. 400 — невалидные
входы, 404 — неизвестная модель или товар, 503 — модель не загружена.

### Docker

```bash
docker build -t retail-recsys .
docker run --rm -p 8000:8000 --env-file .env retail-recsys
# без S3 — смонтировать локальные артефакты:
docker run --rm -p 8000:8000 -v "$PWD/models:/app/models" -v "$PWD/data:/app/data" retail-recsys
```

Модель в образ не запекается: при старте контейнер скачивает её из S3.

---

## Препроцессинг в Airflow (`mlops/airflow`)

DAG `preprocessing_data` ([mlops/airflow/preprocessing_data.py](mlops/airflow/preprocessing_data.py))
берёт сырые данные из `S3_PREFIX/raw_data`, строит признаки тем же кодом из
`src/` и кладёт результат в `S3_PREFIX/processed_data`:

```
check_raw_data → extract → transform → validate (pandera) → load → cleanup
```

* Airflow 2.10 (LocalExecutor + Postgres) запускается через
  [mlops/airflow/docker-compose.yaml](mlops/airflow/docker-compose.yaml).
  Образ собирается из `apache/airflow` с зависимостями `src/`.
  Код `src/` и `config/` монтируется в контейнер.
* **Секреты S3** хранятся в Airflow Connection `s3_storage`. Её создаёт
  `airflow-init` из `.env`, пароль шифруется `AIRFLOW_FERNET_KEY`.
  Webserver и scheduler получают только бакет и префикс.
* Параметр запуска `sample` позволяет прочитать только N строк сырых CSV.
* Запуск DAG ручной (`schedule=None`).

```bash
# на ВМ, из корня репозитория (.env заполнен: S3_*, AWS_*, AIRFLOW_*)
docker compose -f mlops/airflow/docker-compose.yaml --env-file .env up -d --build
# UI: http://<ВМ>:${AIRFLOW_PORT:-8080} → preprocessing_data → Trigger DAG
```

Сырые CSV нужно один раз загрузить в бакет:

```bash
aws --endpoint-url https://storage.yandexcloud.net s3 cp data/raw/ \
  s3://$S3_BUCKET/$S3_PREFIX/raw_data/ --recursive --exclude "*" --include "*.csv"
```

---

## Пайплайн обучения в ClearML (`mlops/clearml`)

[mlops/clearml/clearml_pipeline.py](mlops/clearml/clearml_pipeline.py) — пайплайн `Train catboost model`:

```
train_candidate ×N (параллельно) ──► select_best_model ──► S3 models/ranker.pkl + Model Registry
```

* Кандидаты гиперпараметров задаются в
  [mlops/clearml/search_space.json](mlops/clearml/search_space.json). Каждый
  кандидат переопределяет `model_params` из `config.json`.
* Данные — `S3_PREFIX/processed_data` (результат DAG). Выборка
  стратифицированно делится на train / valid / test.
* Каждый кандидат обучается на train с early stopping по valid и логирует
  метрики valid в ClearML.
* `select_best_model` выбирает кандидата с максимальным **PR-AUC** на valid
  (положительных примеров мало, поэтому PR-AUC, а не accuracy/ROC-AUC).
  Затем он оценивает модель на отложенном test, загружает `ranker.pkl` в
  `S3_PREFIX/models/` и регистрирует модель `ranker` в ClearML Model Registry
  с тегом `best`. Сервис берёт модель из S3.

```bash
uv pip install -r mlops/clearml/requirements.txt
PYTHONPATH=. python mlops/clearml/clearml_pipeline.py --mode local                 # шаги на этой машине
PYTHONPATH=. python mlops/clearml/clearml_pipeline.py --mode remote --queue default # шаги на clearml-agent
PYTHONPATH=. python mlops/clearml/clearml_pipeline.py --mode debug                 # всё в одном процессе
```

Подключение к ClearML задаётся переменными `CLEARML_*` в `.env`.

---

## CI/CD (`.gitlab-ci.yml`)

| Стадия | Джоба | Что делает |
|---|---|---|
| lint | `ruff_lint`, `mypy` | PEP8/pyflakes/isort/bugbear (отчёт Code Quality), типы |
| formating | `ruff_format` | `ruff format --check` (стиль black) |
| test | `pytest` | unit-тесты, JUnit-отчёт в MR |
| build | `build_image` | buildkit (rootless) → `$CI_REGISTRY_IMAGE:<sha>` в GitLab Container Registry, кэш слоёв в registry |
| save | `save_image` | протестированный образ помечается тегом `latest` (default-ветка) или тегом релиза (git tag) |
| deploy | `deploy_vm` | деплой на ВМ по SSH: конфиги + `docker compose up` с образом коммита, затем smoke-check `/health` |

Секреты хранятся в **Settings → CI/CD → Variables** (masked + protected),
в репозитории их нет:

| Переменная | Тип | Назначение |
|---|---|---|
| `VM_HOST`, `VM_USER` | Variable | адрес ВМ и пользователь. Без `VM_HOST` джоба деплоя не запускается |
| `SSH_PRIVATE_KEY` | File | ключ доступа к ВМ |
| `VM_ENV_FILE` | File | содержимое `.env` для ВМ (S3, Grafana) |

---

## Деплой на ВМ (`mlops/deploy/vm`)

[mlops/deploy/vm/docker-compose.yml](mlops/deploy/vm/docker-compose.yml)
поднимает инференс-сервис на порту **8000** вместе со стеком мониторинга.
Модель в образ не запекается: при старте сервис скачивает `ranker.pkl` и
`item_features.parquet` из S3. Последние скачанные файлы хранятся в volume
и служат fallback, если S3 недоступен.

```bash
# из корня репозитория; APP_IMAGE — образ из Registry (пусто — сборка локально)
docker compose -f mlops/deploy/vm/docker-compose.yml --env-file .env up -d
curl "http://127.0.0.1:8000/predict_purchase?itemid=356475&hour=14&weekday=3"
```

---

## Деплой в Kubernetes (`mlops/deploy/k8s`)

| Файл | Что внутри |
|---|---|
| [deployment.yml](mlops/deploy/k8s/deployment.yml) | 2 реплики. **init-контейнер** `fetch-artifacts` (`python -m scripts.fetch_artifacts`) скачивает модель и признаки из S3 в общий `emptyDir`. Основной контейнер работает с локальными файлами без ключей S3. Пробы: startup/liveness `/v2/health/live`, readiness `/v2/health/ready` |
| [service.yml](mlops/deploy/k8s/service.yml) | ClusterIP, порт 8080 → 8000 |
| [deploy.sh](mlops/deploy/k8s/deploy.sh) | создаёт из `.env` Secret `recsys-s3`, ConfigMap `recsys-config` и (опционально) pull-secret реестра, применяет манифесты с образом `K8S_IMAGE` и делает `rollout restart`, чтобы модель скачалась заново |

```bash
./mlops/deploy/k8s/deploy.sh                       # параметры: K8S_NAMESPACE, K8S_IMAGE, S3_*, AWS_* в .env
kubectl -n $K8S_NAMESPACE port-forward svc/recsys-api 8080:8080
curl http://127.0.0.1:8080/v2/health/ready
curl "http://127.0.0.1:8080/v2/models/ranker/infer?itemid=356475&hour=14&weekday=3"
```

Init-контейнер сделан строгим: если S3 не настроен или модели нет, под не
стартует. Это лучше, чем поднять под без модели.

---

## Мониторинг

**Сервис.** `GET /metrics` отдаёт метрики в формате Prometheus
([src/api/metrics.py](src/api/metrics.py)):

| Метрика | Что показывает |
|---|---|
| `http_requests_total{method,handler,status}` | RPS и ошибки по ручкам (handler — шаблон пути) |
| `http_request_duration_seconds` | latency (гистограмма → p50/p95/p99) |
| `http_requests_in_progress` | запросы в обработке |
| `recsys_model_loaded`, `recsys_feature_store_items` | модель загружена, размер feature store |
| `recsys_predictions_total{endpoint}` | число предсказаний |
| `recsys_prediction_score{endpoint}` | **распределение score** (гистограмма вероятностей) |
| `recsys_prediction_errors_total{endpoint,reason}` | `unknown_item`, `model_not_loaded` |
| `recsys_prediction_requests_by_{hour,weekday}_total` | распределение входов: дрейф поведения пользователей |

Каждое предсказание и каждая ошибка пишутся в лог. При `LOG_FORMAT=json`
это одна JSON-строка с полями `event`, `endpoint`, `itemid`, `hour`,
`weekday`, `score`.

**Стек на ВМ** (тот же compose, что и сервис):

| Компонент | Порт | Конфиг |
|---|---|---|
| Prometheus | 9090 | [prometheus.yml](mlops/monitoring/prometheus/prometheus.yml), алерты [alerts.yml](mlops/monitoring/prometheus/alerts.yml) |
| Alertmanager | 9093 | [alertmanager.yml](mlops/monitoring/alertmanager/alertmanager.yml) |
| Loki + Promtail | 3100 | [promtail.yml](mlops/monitoring/promtail/promtail.yml): логи контейнеров с меткой `logging=promtail` |
| Grafana | 3000 | provisioning источников и дашборда [recsys.json](mlops/grafana/dashboards/recsys.json) |

Дашборд **Retail RecSys** содержит:
- состояние сервиса и модели, RPS, долю 5xx, p95;
- RPS по ручкам, latency p50/p95/p99, ошибки по статусу и причине;
- распределение score, его среднее и квантили во времени;
- распределение запросов по часу и дню недели;
- активные алерты;
- логи ошибок и предсказаний из Loki.

**Алерты:**

| Алерт | Условие |
|---|---|
| `RecsysApiDown` | `/metrics` не снимается > 1 мин |
| `RecsysModelNotLoaded` | модель не загружена > 2 мин |
| `RecsysHigh5xxRate` | доля 5xx > 5% за 5 мин |
| `RecsysHighLatencyP95` | p95 инференса > 500 мс |
| `RecsysUnknownItemsRate` | > 20% запросов к неизвестным товарам (устаревший feature store) |
| `RecsysScoreDrift` | средний score за час отличается от суточного > 50% (при трафике > 100 запросов/ч) |

Алерты видны в Prometheus (`/alerts`), Alertmanager и на дашборде. Чтобы
получать их в мессенджер, добавьте receiver в `alertmanager.yml`.

---

## Качество кода

Все настройки лежат в `pyproject.toml`.

```bash
ruff check .            # линтер (pycodestyle, pyflakes, isort, bugbear, pyupgrade)
ruff format --check .   # форматирование в стиле black
mypy                    # статическая типизация (src, scripts, tests)
pytest                  # unit-тесты
```

Тесты:

* `tests/test_main.py`: эндпоинты `/predict_purchase`, `/health` и `/version`.
  Вместо реальной модели подставляется заглушка через `dependency_overrides`
  (`tests/conftest.py`), так что S3 и данные не нужны;
* `tests/test_v2.py`: V2 Inference Protocol (health, метаданные, батч-инференс,
  валидация входов);
* `tests/test_metrics.py`: `/metrics`, счётчики запросов, score, ошибок,
  JSON-логи;
* `tests/test_fetch_artifacts.py`: скрипт init-контейнера;
* `tests/test_storage.py`: работа с S3 (клиент замокан) и локальный fallback;
* `tests/test_selection.py`: разбиение train/valid/test, пространство
  гиперпараметров, выбор лучшей модели, early stopping;
* `tests/test_prepare_data.py`: загрузка, предобработка, признаки, таргет;
* `tests/test_validation.py`: валидация обучающей выборки.

---

## Структура проекта

```
├── config/config.json          # ML-параметры (файлы, признаки, гиперпараметры)
├── mlops/
│   ├── airflow/                # DAG preprocessing_data.py, docker-compose, образ Airflow
│   ├── clearml/                # clearml_pipeline.py, search_space.json
│   ├── deploy/
│   │   ├── vm/                 # docker-compose: сервис + Prometheus/Grafana/Loki
│   │   └── k8s/                # deployment.yml, service.yml, deploy.sh
│   ├── monitoring/             # prometheus.yml, alerts.yml, alertmanager, promtail
│   └── grafana/                # provisioning + дашборд recsys.json
├── scripts/
│   ├── prepare_data.py         # шаг 1: сырые данные -> признаки
│   ├── train_model.py          # шаг 2: обучение и оценка модели
│   └── fetch_artifacts.py      # init-контейнер k8s: модель и признаки из S3
├── src/
│   ├── main.py                 # FastAPI-сервис
│   ├── api/                    # V2 Inference Protocol, метрики Prometheus, инференс
│   ├── data/                   # загрузка, предобработка, валидация
│   ├── features/               # сборка признаков и таргета
│   ├── models/                 # обучение, метрики, загрузка артефактов для сервиса
│   ├── schemas/                # pydantic (API, конфиг) и pandera (данные)
│   └── utils/                  # settings (.env), storage (S3 + fallback), config, logging
├── tests/                      # unit-тесты
├── .env.example                # шаблон настроек окружения
├── .gitlab-ci.yml              # CI/CD
├── Dockerfile
├── pyproject.toml              # зависимости, ruff, mypy, pytest
├── requirements.txt            # runtime-зависимости
└── requirements-dev.txt        # + ruff, mypy, pytest
```

---

## Статус

- [x] Структура проекта, слой настроек `.env`, S3-хранилище с локальным fallback
- [x] FastAPI-сервис, unit-тесты API, ruff / mypy / pytest без ошибок
- [x] Dockerfile
- [x] CI/CD: lint, formating, test, build (buildkit -> GitLab Registry), save
- [x] Препроцессинг в Airflow (`mlops/airflow/preprocessing_data.py`)
- [x] Пайплайн обучения и выбора модели в ClearML (`mlops/clearml/clearml_pipeline.py`)
- [x] Деплой на ВМ (docker-compose, стадия deploy в CI)
- [x] Мониторинг: `/metrics`, Prometheus, Grafana, Loki, алерты
- [x] (доп.) Деплой в Kubernetes: Deployment с init-контейнером + Service, V2 API
- [ ] Прогон на реальной инфраструктуре (GitLab CI, S3, ВМ, ClearML, k8s) и скриншоты Grafana / Airflow / ClearML
