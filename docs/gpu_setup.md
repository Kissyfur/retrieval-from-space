# GPU Setup On Windows

There are two practical TensorFlow GPU routes for this project.

## Native Windows GPU

Use this when you want TensorFlow to see the GPU from normal Windows Python.

Keep the old pins:

```text
keras~=2.10.0
tensorflow~=2.10.1
```

Use Python 3.9 or 3.10, then install the CUDA stack expected by TensorFlow 2.10:

```bash
conda create --name rfs-tf210 python=3.10
conda activate rfs-tf210
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install -r requirements.txt
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

This remains the native Windows CUDA path because TensorFlow 2.10 was the last release with native Windows GPU support.

## WSL2 GPU

Use this when you can run the project inside WSL2. This is the modern official route for TensorFlow GPU on a Windows machine.

Requirements:

- Windows 10 21H2 / build 19044 or newer, or Windows 11
- NVIDIA driver with WSL support
- WSL2 Linux environment

Inside WSL2:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[wsl2-gpu]"
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

If you need the exact Windows-native GPU behavior, prefer the native Windows GPU environment above.

## DirectML

TensorFlow DirectML is not the main recommendation for this project. Microsoft's TensorFlow DirectML plugin repository says development is paused, and the repository describes the plugin as not production supported. It may be useful for experiments on non-NVIDIA hardware, but it is not a stable default for this toolkit.

## Sources

- TensorFlow pip install guide: https://www.tensorflow.org/install/pip
- TensorFlow DirectML plugin repository: https://github.com/microsoft/tensorflow-directml-plugin
