FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
# CPU-версия torch — образ в ~5 раз меньше
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY app.py .
COPY weights/ weights/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
