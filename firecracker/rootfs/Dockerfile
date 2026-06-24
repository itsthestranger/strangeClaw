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

# Copy only runtime assets needed in the guest. The direct LiteLLM client
# (agent/llm.py) is intentionally host-only and is not copied into the guest.
RUN mkdir -p agent
COPY agent/__init__.py agent/agent.py agent/broker_client.py agent/llm_proxy.py agent/llm_types.py \
     agent/protocol.py agent/skills.py agent/subagents.py agent/tools.py agent/transport.py ./agent/
COPY skills ./skills
COPY firecracker/rootfs/entrypoint.sh /tmp/entrypoint.sh

# Guest runtime dependencies are intentionally minimal. HTTP and LLM calls run
# through host services, so no HTTP client, LiteLLM, or provider SDK is installed
# in the guest venv.
RUN python3.11 -m venv /opt/strangeclaw/.venv \
    && /opt/strangeclaw/.venv/bin/pip install --upgrade pip \
    && /opt/strangeclaw/.venv/bin/pip install --no-cache-dir 'PyYAML>=6.0.1' \
    && install -m 0755 /tmp/entrypoint.sh /sbin/entrypoint.sh \
    && ln -sf /usr/bin/tini /sbin/tini \
    && rm -f /sbin/init \
    && printf '#!/bin/sh\nexec /sbin/tini -- /sbin/entrypoint.sh\n' > /sbin/init \
    && chmod 0755 /sbin/init \
    && mkdir -p /run/strangeclaw /output
