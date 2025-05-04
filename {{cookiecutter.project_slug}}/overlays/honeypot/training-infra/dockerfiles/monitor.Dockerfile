FROM python:3.12-slim

WORKDIR /app

RUN pip install fastapi==0.104.1 uvicorn==0.24.0 minio==7.2.0

COPY monitor.py /app/

EXPOSE 8080

CMD ["uvicorn", "monitor:app", "--host", "0.0.0.0", "--port", "8080"]