# ============================================================
# TN-NTN LightGBM xApp — Multi-stage Docker Build
# Multi-orbit broadband handover with ensemble prediction
# ============================================================

# ----- Stage 1: Builder (compile C extensions) -----
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install \
        --timeout 300 --retries 3 \
        -r requirements.txt

# ----- Stage 2: Runtime -----
FROM python:3.11-slim AS runtime

# libgomp1: OpenMP runtime for LightGBM/XGBoost/CatBoost parallel inference
# curl: Docker health-check probes
# wget/ca-certificates: RMR C library download
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

# Install RMR shared library AFTER builder copy (builder overwrites /usr/local)
RUN wget -q https://packagecloud.io/o-ran-sc/release/packages/debian/stretch/rmr_4.9.4_amd64.deb/download.deb \
        -O /tmp/rmr.deb \
    && dpkg -i /tmp/rmr.deb \
    && rm -f /tmp/rmr.deb \
    && ldconfig

WORKDIR /app

# Application source
COPY src/        ./src/
COPY schemas/    ./schemas/

# Pre-populate models for fast startup
COPY models/     ./models/

# xApp descriptors (O-RAN onboarding)
COPY config-file.json xapp-descriptor.yaml ./

COPY requirements.txt ./

# Non-root user
RUN groupadd -r tnntn && useradd -r -g tnntn -d /app -s /sbin/nologin tnntn \
    && chown -R tnntn:tnntn /app
USER tnntn

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    XAPP_PORT=8080 \
    MODEL_PATH=/app/models/multi_orbit_ensemble.pkl \
    DBAAS_SERVICE_HOST=dbaas \
    RMR_SRC_ID=tn-ntn-handover

EXPOSE 8080 4560 4561

CMD ["python", "-m", "src.main", \
     "--host", "0.0.0.0", \
     "--port", "8080"]
