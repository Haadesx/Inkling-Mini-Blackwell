FROM nvidia/cuda@sha256:d266e59b88c295bc5fa0e4cef9064eaff84939381b09e2c3d76a5532a303e42d
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates cmake g++ git make libcurl4-openssl-dev libdw1t64 \
    && rm -rf /var/lib/apt/lists/*
