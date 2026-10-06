#!/usr/bin/env bash
set -Eeuo pipefail

# Install the runtime for this checkout and start the Gradio application.
# The script is self-contained and does not depend on the directory from
# which it is called.
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${STROKE2SKETCH_ENV:-stroke2sketch-runtime}"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
MODEL_DIR="${ROOT_DIR}/model/stable-diffusion-v1-5"
MODEL_ID="${STROKE2SKETCH_MODEL_ID:-stable-diffusion-v1-5/stable-diffusion-v1-5}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PORT="${GRADIO_SERVER_PORT:-7860}"
START=1
BACKGROUND=0
WITH_U2NET=0

usage() {
    cat <<EOF
用法: $(basename "$0") [选项]

首次运行会安装 Miniconda/Conda 环境、Python 依赖并下载 Stable Diffusion
模型，然后启动网页。默认启动地址为 http://0.0.0.0:${PORT}。

选项:
  --no-start       只安装和检查，不启动网页
  --background     后台启动网页，日志写入 output/app.log
  --with-u2net     缺少 U2Net 时从 U2NET_URL 下载权重
  --port PORT      设置网页端口（默认: ${PORT}）
  -h, --help       显示帮助
EOF
}

while (($#)); do
    case "$1" in
        --no-start) START=0 ;;
        --background) BACKGROUND=1 ;;
        --with-u2net) WITH_U2NET=1 ;;
        --port)
            (($# >= 2)) || { echo "错误：--port 需要端口号" >&2; exit 2; }
            PORT="$2"
            shift
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "错误：未知选项 $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

die() {
    echo "错误：$*" >&2
    exit 1
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

download_file() {
    local url="$1"
    local output="$2"
    if command_exists curl; then
        curl -fL --retry 3 --retry-delay 2 "$url" -o "$output"
    elif command_exists wget; then
        wget -O "$output" "$url"
    else
        die "需要 curl 或 wget 才能下载文件"
    fi
}

[[ "$(uname -s)" == "Linux" ]] || die "此脚本目前只支持 Linux"
[[ "$(uname -m)" == "x86_64" ]] || die "此脚本需要 Linux x86_64 的 CUDA 版 PyTorch"
[[ "$PORT" =~ ^[0-9]+$ ]] && ((10#$PORT > 0 && 10#$PORT <= 65535)) || die "端口需要是 1 到 65535 的整数"
command_exists nvidia-smi || die "请先安装 NVIDIA 驱动，确保 nvidia-smi 可用"
nvidia-smi -L

# Prefer an existing Conda installation. Otherwise install Miniconda under
# $CONDA_ROOT, which does not require root privileges.
if command_exists conda; then
    CONDA_BASE="$(conda info --base)"
    CONDA_BIN="$CONDA_BASE/bin/conda"
elif [[ -x "$CONDA_ROOT/bin/conda" ]]; then
    CONDA_BIN="$CONDA_ROOT/bin/conda"
    CONDA_BASE="$CONDA_ROOT"
else
    command_exists curl || command_exists wget || die "找不到 curl 或 wget，无法下载 Miniconda"
    mkdir -p "$(dirname "$CONDA_ROOT")"
    installer="$(mktemp)"
    trap 'rm -f "$installer"' EXIT
    echo "下载 Miniconda 到 $CONDA_ROOT ..."
    download_file "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" "$installer"
    bash "$installer" -b -p "$CONDA_ROOT"
    rm -f "$installer"
    trap - EXIT
    CONDA_BIN="$CONDA_ROOT/bin/conda"
    CONDA_BASE="$CONDA_ROOT"
fi

# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! "$CONDA_BIN" run -n "$ENV_NAME" python -c 'import sys' >/dev/null 2>&1; then
    echo "创建 Conda 环境 $ENV_NAME ..."
    "$CONDA_BIN" create -n "$ENV_NAME" python=3.8 pip -y --override-channels -c conda-forge
fi
conda activate "$ENV_NAME"
PYTHON="$(command -v python)"

echo "安装 Python 依赖 ..."
"$PYTHON" -m pip install -q pip==25.0.1
"$PYTHON" -m pip install -q \
    torch==2.0.1 torchvision==0.15.2 \
    numpy==1.24.1 Pillow==10.1.0 opencv-python-headless==4.8.1.78 \
    diffusers==0.19.3 transformers==4.30.2 huggingface-hub==0.19.4 \
    accelerate==0.20.3 safetensors==0.5.3 \
    gradio==4.44.1 fastapi==0.116.1 pydantic==2.10.6 \
    pyrallis==0.3.1 kornia==0.7.3 kornia-rs==0.1.10 \
    scikit-image==0.21.0 scikit-learn==1.3.0 scipy==1.10.1 \
    nltk==3.8.1 tqdm==4.64.1

"$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA 不可用：请检查 NVIDIA 驱动和 CUDA 版 PyTorch")
torch.zeros(1, device="cuda")
print("CUDA:", torch.cuda.get_device_name(0))
PY

required_model_files=(
    model_index.json
    scheduler/scheduler_config.json
    text_encoder/config.json
    tokenizer/merges.txt
    tokenizer/vocab.json
    tokenizer/tokenizer_config.json
    tokenizer/special_tokens_map.json
    unet/config.json
    vae/config.json
    feature_extractor/preprocessor_config.json
)
for weights in text_encoder/model unet/diffusion_pytorch_model vae/diffusion_pytorch_model; do
    if [[ ! -s "$MODEL_DIR/$weights.safetensors" && ! -s "$MODEL_DIR/$weights.bin" ]]; then
        required_model_files+=("$weights.safetensors")
    fi
done
model_ready=1
for file in "${required_model_files[@]}"; do
    [[ -s "$MODEL_DIR/$file" ]] || { model_ready=0; break; }
done

if ((model_ready == 0)); then
    echo "下载 Stable Diffusion 模型 $MODEL_ID ..."
    mkdir -p "$MODEL_DIR"
    HF_ENDPOINT="$HF_ENDPOINT" "$PYTHON" - "$MODEL_ID" "$MODEL_DIR" "${required_model_files[@]}" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

for filename in sys.argv[3:]:
    target = Path(sys.argv[2]) / filename
    if target.is_file() and target.stat().st_size:
        continue
    hf_hub_download(
        repo_id=sys.argv[1], filename=filename, local_dir=sys.argv[2],
        local_dir_use_symlinks=False, resume_download=True,
    )
PY
else
    echo "Stable Diffusion 模型已存在，跳过下载。"
fi

if ((WITH_U2NET == 1)); then
    if [[ ! -s "$ROOT_DIR/U2Net_/saved_models/u2net.pth" ]]; then
        [[ -n "${U2NET_URL:-}" ]] || die "--with-u2net 需要设置可直接下载权重的 U2NET_URL"
        mkdir -p "$ROOT_DIR/U2Net_/saved_models"
        echo "下载 U2Net 权重 ..."
        u2net_tmp="$ROOT_DIR/U2Net_/saved_models/u2net.pth.tmp"
        download_file "$U2NET_URL" "$u2net_tmp"
        mv "$u2net_tmp" "$ROOT_DIR/U2Net_/saved_models/u2net.pth"
    else
        echo "U2Net 权重已存在，跳过下载。"
    fi
fi

cd "$ROOT_DIR"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON" -c 'import run, app'
echo "环境、模型和网页模块检查完成。"

((START == 1)) || exit 0
export GRADIO_SERVER_PORT="$PORT"
export GRADIO_NUM_PORTS=1
export HF_ENDPOINT
if ((BACKGROUND == 1)); then
    mkdir -p "$ROOT_DIR/output"
    log_file="$ROOT_DIR/output/app.log"
    nohup "$PYTHON" -u "$ROOT_DIR/app.py" >"$log_file" 2>&1 &
    echo "网页已在后台启动，PID: $!"
    echo "日志: $log_file"
    echo "访问: http://<服务器IP>:$PORT"
else
    echo "启动网页：http://0.0.0.0:$PORT"
    exec "$PYTHON" "$ROOT_DIR/app.py"
fi
