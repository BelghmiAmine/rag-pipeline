#####################################
# RCP CaaS requirement (Image)
#####################################
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.11 + pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python/pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

#####################################
# RCP CaaS requirement (Storage)
#####################################
ARG LDAP_USERNAME
ARG LDAP_UID
ARG LDAP_GROUPNAME
ARG LDAP_GID
RUN groupadd ${LDAP_GROUPNAME} --gid ${LDAP_GID}
RUN useradd -m -s /bin/bash -g ${LDAP_GROUPNAME} -u ${LDAP_UID} ${LDAP_USERNAME}
#####################################

# --- Install torch BEFORE copying code ---
# This layer is cached permanently. Changing .py files or requirements.txt
# will NOT re-trigger this download ever again.
RUN pip install --no-cache-dir torch==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# --- Copy and install requirements separately from code ---
# Changing requirements.txt rebuilds from here, but NOT the torch layer above.
# Changing .py files only rebuilds from the COPY below, not pip installs.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu124

# --- Verify imports work at build time ---
RUN python3 -c "import datasets; import pandas; import torch; import numpy; import sentence_transformers; import openai; import langchain_openai; print('All imports OK')"

# --- Copy code last ---
# This is the only layer that changes when you edit your .py files.
RUN mkdir -p /home/${LDAP_USERNAME}
COPY ./ /home/${LDAP_USERNAME}
RUN chown -R ${LDAP_USERNAME}:${LDAP_GROUPNAME} /home/${LDAP_USERNAME}

WORKDIR /home/${LDAP_USERNAME}
USER ${LDAP_USERNAME}