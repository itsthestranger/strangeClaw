FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      bash \
      ca-certificates \
      curl \
      dirmngr \
      gnupg \
      iproute2 \
      software-properties-common \
      tini \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      python3.11 \
      python3.11-venv \
      python3-pip \
    && python3.11 -c "import sys; assert sys.version_info.releaselevel == 'final', sys.version" \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/strangeclaw

# Copy only runtime assets needed in the guest.
COPY pyproject.toml README.md ./
COPY config.py coordinator.py main.py session.py ./
COPY agent ./agent
COPY adapters ./adapters
COPY sandbox ./sandbox
COPY skills ./skills
COPY firecracker/rootfs/entrypoint.sh /tmp/entrypoint.sh

RUN python3.11 -m venv /opt/strangeclaw/.venv \
    && /opt/strangeclaw/.venv/bin/pip install --upgrade pip \
    && /opt/strangeclaw/.venv/bin/pip install --no-cache-dir . \
    && /opt/strangeclaw/.venv/bin/pip uninstall -y trafilatura \
    && install -m 0755 /tmp/entrypoint.sh /sbin/entrypoint.sh \
    && ln -sf /usr/bin/tini /sbin/tini \
    && rm -f /sbin/init \
    && printf '#!/bin/sh\nexec /sbin/tini -- /sbin/entrypoint.sh\n' > /sbin/init \
    && chmod 0755 /sbin/init \
    && mkdir -p /run/strangeclaw /output
