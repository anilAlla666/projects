#!/bin/sh
# One-time setup of a Lambda GPU instance: uv, Python 3.12, PyTorch (CUDA 12.8), flash-linear-attention.
# Usage, from the Mac: HOST=ubuntu@<ip> sh scripts/setup_gpu.sh
set -e
HOST=${HOST:?set HOST=ubuntu@<ip>}
sh "$(dirname "$0")/sync.sh"
ssh -o StrictHostKeyChecking=accept-new "$HOST" 'set -e
python3 -m pip install -q --user uv
export PATH=$HOME/.local/bin:$PATH
uv python install 3.12 >/dev/null
[ -d ~/vmenv ] || uv venv -q ~/vmenv --python 3.12
VIRTUAL_ENV=~/vmenv uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128
VIRTUAL_ENV=~/vmenv uv pip install -q flash-linear-attention pytest numpy
~/vmenv/bin/python -c "import torch, fla; print(\"torch\", torch.__version__, \"fla\", fla.__version__, \"gpus\", torch.cuda.device_count(), torch.cuda.get_device_name())"
nproc; free -g | sed -n 2p; df -h / | tail -1'
