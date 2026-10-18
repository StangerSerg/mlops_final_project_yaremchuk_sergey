FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN uv pip install --no-cache --system -r requirements.txt
COPY --from=builder /app/model.pkl .
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 