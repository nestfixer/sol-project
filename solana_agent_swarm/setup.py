from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="solana_agent_swarm",
    version="0.1.0",
    description="AI agent swarm for Solana token analysis",
    author="Solana Token Analysis Team",
    author_email="example@example.com",
    packages=find_packages(),
    package_data={
        '': ['*.json'],
    },
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'solana-swarm=src.main:main',
            'solana-cli=src.cli:main',
        ],
    },
    python_requires='>=3.8',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
