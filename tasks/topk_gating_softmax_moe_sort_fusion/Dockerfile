# Installs this branch's aiter commit into the vLLM ROCm base image used for
# testing topk_gating_softmax_moe_sort_fusion.
#
# Build:
#   docker build -f tasks/topk_gating_softmax_moe_sort_fusion/Dockerfile \
#     --build-arg BASE_IMAGE=vllm/vllm-openai-rocm:nightly-rocm100-ac9126e58aa7bbab1856ba6593ba4d5003fea516 \
#     --build-arg AITER_COMMIT=2ed2f032a \
#     -t vllm/vllm-openai-rocm:nightly-rocm100-ac9126e58aa7bbab1856ba6593ba4d5003fea516_aiter_2ed2f032a .

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG AITER_REPO=https://github.com/samutamm/aiter.git
ARG AITER_BRANCH=topk_gating_softmax_moe_sort_fusion
ARG AITER_COMMIT=2ed2f032a
# Bust the git-clone layer's cache -- pass a fresh value (e.g. `date +%s`)
# on every build so the image always picks up the commit's current content
# instead of silently reusing a stale cached clone.
ARG CACHEBUST=0

# The base image ships its own non-editable amd-aiter install under
# dist-packages/aiter; `setup.py develop`'s easy-install.pth entry is
# appended *after* dist-packages on sys.path, so without removing it first
# `import aiter` would keep resolving to the old baked-in copy instead of
# our /aiter checkout.
RUN pip uninstall -y amd-aiter || true \
    && rm -rf /usr/local/lib/python3.12/dist-packages/aiter

RUN echo "cachebust=${CACHEBUST}" \
    && git clone --recursive --branch ${AITER_BRANCH} ${AITER_REPO} /aiter \
    && cd /aiter \
    && git checkout ${AITER_COMMIT} \
    && git submodule update --init --recursive \
    && python3 setup.py develop
RUN pip install --no-cache-dir tabulate
