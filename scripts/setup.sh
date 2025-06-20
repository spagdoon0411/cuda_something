pip install nvcc4jupyter
export CUDA_DISABLE_PTX_JIT=1
echo "Toolkit/runtime: $(nvcc --version)"
nvidia-smi
