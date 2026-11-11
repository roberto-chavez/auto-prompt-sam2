# Gradio on Docker Spaces. No model weights are fetched during the build.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SAM2_BUILD_CUDA=0 \
    GRADIO_ANALYTICS_ENABLED=False \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y --no-install-recommends git libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
COPY requirements-demo.txt requirements-demo-inference.txt ./
RUN pip install --no-cache-dir -r requirements-demo-inference.txt \
    && pip install --no-cache-dir setuptools==80.9.0 wheel==0.45.1 \
    && pip install --no-build-isolation --no-deps \
       git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4 \
    && pip check

COPY --chown=user:user app.py ./
COPY --chown=user:user src ./src
COPY --chown=user:user configs/demo.json ./configs/demo.json
COPY --chown=user:user docs/project/evidence/stage2/official_preprocessing.json ./docs/project/evidence/stage2/official_preprocessing.json
COPY --chown=user:user docs/project/evidence/e2/colab_run_20261030T013250Z/run/report.json ./docs/project/evidence/e2/colab_run_20261030T013250Z/run/report.json
RUN chown user:user /home/user/app
USER user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    DEMO_CACHE_DIR=/home/user/.cache/demo \
    DEMO_MODE=preview
EXPOSE 7860
CMD ["python", "app.py"]
