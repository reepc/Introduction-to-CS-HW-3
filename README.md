Install Llama-cpp-python
github: https://github.com/abetlen/llama-cpp-python

If you are going to use GPU to run the llama-cpp-python:
With CUDA:
CMAKE_ARGS="-DGGML_CUDA=on -DLLAVA_BUILD=off" pip install llama-cpp-python --verbose -U

With JUST CPU:
CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python

Others:
https://github.com/abetlen/llama-cpp-python#Installation

Install pytorch here:
https://pytorch.org/get-started/locally/
Choose the option that is fit for yourself