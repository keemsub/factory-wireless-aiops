FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir numpy scikit-learn "kafka-python>=2.0.3,<3" prometheus-client
COPY src/ /app/src/
COPY artifacts/ /app/artifacts/
WORKDIR /app/src
ENV PYTHONUNBUFFERED=1
