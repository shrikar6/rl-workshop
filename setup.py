from setuptools import setup, find_packages

setup(
    name="rl-workshop",
    version="0.1.0",
    description="A modular reinforcement learning framework for plug-and-play experimentation and personal education",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "jax[cpu]>=0.7.0,<0.8.0",
        "optax>=0.2.0,<0.3.0",
        "gymnasium[classic-control]>=1.0.0,<2.0.0",
        "matplotlib>=3.0.0,<4.0.0",
        "imageio[ffmpeg]>=2.0.0,<3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0,<9.0.0",
            "pytest-cov>=6.0.0,<7.0.0",
        ]
    }
)