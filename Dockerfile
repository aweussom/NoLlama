# NoLlama in a container. Model-free by design: weights are bind-mounted
# read-only at run time, never baked in and never downloaded here.
#
# GPU and CPU only. There is no NPU path — the NPU is not exposed to WSL 2
# or to Linux containers on Windows (DOCKER-INSTALL.md, Track B).
#
# Measured on WSL 2 + Docker Desktop with an Arc Pro B60; NOT yet measured on
# native Linux with /dev/dri. See DOCKER-INSTALL.md for what that means.
FROM ubuntu:24.04

# Intel GPU userspace, pinned to an upstream *release* (leading edge, not
# bleeding edge — docs/dev/runtime-stacks.md). Distro packages are not an
# option here: Ubuntu 24.04 ships NEO 24.x, which predates Battlemage and
# enumerates no GPU at all rather than failing loudly (TODONT.md).
ARG NEO=26.31.39395.13
ARG IGC=2.40.13
ARG IGC_BUILD=22418
# libigdgmm is versioned independently and shipped as an asset of each NEO
# release — move it with NEO, not on its own.
ARG GMM=22.10.0

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv curl ca-certificates \
      ocl-icd-libopencl1 libtbb12 libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /tmp/neo && cd /tmp/neo && \
    IGC_URL=https://github.com/intel/intel-graphics-compiler/releases/download/v${IGC} && \
    curl -fsSLO ${IGC_URL}/intel-igc-core-2_${IGC}%2B${IGC_BUILD}_amd64.deb && \
    curl -fsSLO ${IGC_URL}/intel-igc-opencl-2_${IGC}%2B${IGC_BUILD}_amd64.deb && \
    NEO_URL=https://github.com/intel/compute-runtime/releases/download/${NEO} && \
    curl -fsSLO ${NEO_URL}/intel-opencl-icd_${NEO}-0_amd64.deb && \
    curl -fsSLO ${NEO_URL}/libze-intel-gpu1_${NEO}-0_amd64.deb && \
    curl -fsSLO ${NEO_URL}/intel-ocloc_${NEO}-0_amd64.deb && \
    curl -fsSLO ${NEO_URL}/libigdgmm12_${GMM}_amd64.deb && \
    dpkg -i *.deb && rm -rf /tmp/neo

# Serving dependencies only. requirements.txt is deliberately NOT used: it
# also carries optimum-intel/transformers, which exist for model *export* —
# and nothing is ever exported or downloaded inside a container.
COPY requirements-container.txt /tmp/requirements-container.txt
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements-container.txt && \
    rm /tmp/requirements-container.txt
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /app
COPY nollama.py /app/nollama.py
COPY templates /app/templates
COPY static /app/static

# 8000 = OpenAI API + web UI, 11434 = Ollama API. Publishing only 8000
# silently breaks every Ollama client, so compose publishes both.
EXPOSE 8000 11434

# /state is where prewarm-<port>.json belongs when --idle-timeout 0 is used:
# the default location is /app, which lives in the container's writable layer
# and is thrown away by `docker run --rm`.
VOLUME /state

ENTRYPOINT ["python3", "/app/nollama.py"]
CMD ["--help"]
