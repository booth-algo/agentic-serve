from setuptools import setup, find_packages

setup(
    name="llmcompass",
    version="0.1.0",
    description="LLM Compass - Distributed Mobile Inference",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "scalesim",
        "numpy",
        "torch",
        "scikit-learn",
        "joblib",
    ],
)

