#!/bin/bash
set -e

# Kill any stale uvicorn/streamlit on these ports
for p in 8000 8501; do
  pid=$(lsof -ti tcp:"$p" 2>/dev/null || true)
  if [ -n "$pid" ]; then
    kill -9 $pid || true
  fi
done

source ~/miniconda3/etc/profile.d/conda.sh
conda activate airflow_env

airflow scheduler &
airflow webserver &

mlflow server &

echo "Starting FastAPI on :8000 ..."
uvicorn src.data.fastapi_deployment:app --host 0.0.0.0 --port 8000 &
FPID=$!
# Wait for API to be ready
for i in $(seq 1 120); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "FastAPI is ready"
    break
  fi
  sleep 1
done

echo "Starting Streamlit on :8501 ..."
streamlit run src/data/app.py --server.port 8501
