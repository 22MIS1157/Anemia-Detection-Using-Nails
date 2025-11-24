from setuptools import setup, find_packages

setup(
    name="anemia-detection",
    version="1.0.0",
    description="Non-Invasive Anemia Detection from Fingernail Images using Deep Learning",
    author="22MIS1157",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torchmetrics>=1.0.0",
        "timm>=0.9.0",
        "tensorflow>=2.12.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "scikit-learn>=1.3.0",
        "scikit-image>=0.21.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "pennylane>=0.36.0",
        "PyYAML>=6.0"
    ],
)
