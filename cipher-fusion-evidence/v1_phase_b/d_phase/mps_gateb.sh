set -e
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
echo "=== starting MPS control daemon ==="
nvidia-cuda-mps-control -d && echo "MPS daemon launched"
sleep 2
echo "ping_status:"; echo "get_default_active_thread_percentage" | nvidia-cuda-mps-control 2>&1 || true
echo "=== running EXACT Gate B under MPS (green-ctx + MPS shared context) ==="
python3 /home/ubuntu/cipher-fusion-evidence/v1_phase_b/d_phase/d8_ri1_gateb.py
echo "=== MPS server list (a server = clients connected to MPS) ==="
echo "get_server_list" | nvidia-cuda-mps-control 2>&1 || true
echo "=== shutdown MPS ==="
echo quit | nvidia-cuda-mps-control 2>&1 || true
echo "=== MPS_GATEB_DONE ==="
