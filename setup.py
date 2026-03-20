from setuptools import setup, find_packages

setup(
    name="edge-tpu-v5-benchmark-suite",
    version="0.1.0",
    description="Benchmark suite for Google Edge TPU v5 — 50 TOPS/W efficiency target",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Daniel Schmidt",
    python_requires=">=3.8",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[],
    extras_require={
        "dev": ["pytest>=7.0"],
    },
    entry_points={
        "console_scripts": [
            "edge-tpu-v5-bench=edge_tpu_v5_bench.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
