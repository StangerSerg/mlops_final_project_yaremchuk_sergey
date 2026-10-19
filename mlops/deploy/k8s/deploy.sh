#!/usr/bin/env bash
# Деплой инференс-сервиса в Kubernetes.
#
#   ./mlops/deploy/k8s/deploy.sh [путь к .env]   (по умолчанию .env в корне)
#
# Из .env берутся namespace, образ и параметры S3. Создаются (или
# обновляются) Secret recsys-s3, ConfigMap recsys-config и, если задан
# REGISTRY_USER, Secret gitlab-registry для pull образа. Затем
# применяются deployment.yml / service.yml и делается rollout restart,
# чтобы init-контейнер заново скачал модель из S3.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${1:-$DIR/../../../.env}"

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

: "${K8S_NAMESPACE:?set K8S_NAMESPACE in .env}"
: "${K8S_IMAGE:?set K8S_IMAGE in .env}"
: "${S3_BUCKET:?set S3_BUCKET in .env}"
: "${AWS_ACCESS_KEY_ID:?set AWS_ACCESS_KEY_ID in .env}"
: "${AWS_SECRET_ACCESS_KEY:?set AWS_SECRET_ACCESS_KEY in .env}"

kc() { kubectl -n "$K8S_NAMESPACE" "$@"; }

kc create secret generic recsys-s3 \
  --from-literal=AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  --from-literal=AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
  --dry-run=client -o yaml | kc apply -f -

if [ -n "${REGISTRY_USER:-}" ]; then
  kc create secret docker-registry gitlab-registry \
    --docker-server="${REGISTRY:?set REGISTRY in .env}" \
    --docker-username="$REGISTRY_USER" \
    --docker-password="${REGISTRY_PASSWORD:?set REGISTRY_PASSWORD in .env}" \
    --dry-run=client -o yaml | kc apply -f -
fi

kc create configmap recsys-config \
  --from-literal=S3_ENDPOINT_URL="${S3_ENDPOINT_URL:-https://storage.yandexcloud.net}" \
  --from-literal=S3_REGION="${S3_REGION:-ru-central1}" \
  --from-literal=S3_BUCKET="$S3_BUCKET" \
  --from-literal=S3_PREFIX="${S3_PREFIX:-yaremchuk_sergey_project}" \
  --from-literal=S3_PROCESSED_DIR="${S3_PROCESSED_DIR:-processed_data}" \
  --from-literal=S3_MODELS_DIR="${S3_MODELS_DIR:-models}" \
  --from-literal=MODEL_S3_KEY="${MODEL_S3_KEY:-}" \
  --from-literal=MODEL_NAME="${MODEL_NAME:-ranker}" \
  --from-literal=MODEL_VERSION="${MODEL_VERSION:-1}" \
  --dry-run=client -o yaml | kc apply -f -

sed "s|__IMAGE__|$K8S_IMAGE|g" "$DIR/deployment.yml" | kc apply -f -
kc apply -f "$DIR/service.yml"

kc rollout restart deployment/recsys-api
kc rollout status deployment/recsys-api --timeout=300s

echo
echo "Проверка:"
echo "  kubectl -n $K8S_NAMESPACE port-forward svc/recsys-api 8080:8080"
echo "  curl http://127.0.0.1:8080/v2/health/ready"
echo "  curl \"http://127.0.0.1:8080/v2/models/${MODEL_NAME:-ranker}/infer?itemid=356475&hour=14&weekday=3\""
