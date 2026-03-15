ARG ISAACSIM_BASE_IMAGE=nvcr.io/nvidia/isaac-sim:5.1.0
FROM ${ISAACSIM_BASE_IMAGE}

SHELL ["/bin/bash", "-lc"]

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
ENV UV_SYSTEM_PYTHON=1

WORKDIR /workspace
COPY pyproject.toml README.md /workspace/
COPY src /workspace/src
RUN uv pip install --system -e .

ENV ACCEPT_EULA=Y
ENV OMNI_KIT_ACCEPT_EULA=YES
CMD ["bash"]
