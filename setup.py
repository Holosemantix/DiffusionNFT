from setuptools import setup, find_packages

setup(
    name="diffusion-nft",
    version="0.0.1",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        # Core ML (pinned to NPU pre-installed versions)
        "torch==2.1.0",
        "torchvision==0.16.0",
        "transformers>=4.40.0",
        "accelerate>=1.0.0",
        "diffusers>=0.30.0",

        # Scientific computing (compatible with Python 3.10 / NPU env)
        "numpy>=1.23.0",
        "pandas>=1.3.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.5.0",
        "scikit-image>=0.20.0",

        # Vision / data
        "albumentations>=1.3.0",
        "opencv-python>=4.8.0",
        "pillow>=10.0.0",

        # Utils
        "tqdm>=4.66.0",
        "wandb>=0.18.0",
        "pydantic>=2.0.0",
        "requests",
        "matplotlib>=3.7.0",

        # Prevent protobuf from being upgraded to 4.x by sub-dependencies
        "protobuf>=3.20.2,<4",

        # PEFT / HF
        "peft>=0.7.0",

        # Serving / async
        "aiohttp>=3.11.0",
        "fastapi>=0.115.0",
        "uvicorn>=0.34.0",

        # HF ecosystem
        "huggingface-hub>=0.26.0",
        "datasets>=3.0.0",
        "tokenizers>=0.20.0",

        # Others
        "einops>=0.8.0",
        "absl-py",
        "ml_collections",
        "sentencepiece",
    ],
        # NOTE for NPU users:
        #   NPU environments ship most dependencies pre-installed.
        #   Run `pip install -e . --no-deps` to avoid pip upgrading
        #   torch/protobuf/etc. and breaking torch-npu / modelarts.
    extras_require={
        "dev": [
            "ipython>=8.18.0",
            "black>=24.0.0",
            "pytest>=7.4.0",
        ],
        "face": [
            "facenet-pytorch",
            "mediapipe",
        ],
        "flux2": [
            "flux2 @ git+https://github.com/black-forest-labs/flux2.git",
        ],
        "cuda": [
            # CUDA-only packages; not required on NPU
            "flash-attn>=2.7.0",
            "deepspeed>=0.16.0",
            "bitsandbytes>=0.45.0",
            "nvidia-ml-py>=12.570.0",
            "xformers",
        ],
    },
)
