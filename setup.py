from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="trace-clean",
    version="1.0.0",
    author="Jonathan Haas",
    author_email="jonathan@haas.holdings",
    description="A lightweight CLI tool that provides concise summaries of raw stack traces",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/haasonsaas/trace-clean",
    packages=["trace_clean_pkg"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Debuggers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "openai>=1.0.0",
        "rich>=13.0.0",
        "pyyaml>=6.0",
        "requests>=2.28.0",
    ],
    entry_points={
        "console_scripts": [
            "trace-clean=trace_clean_pkg.cli:main",
        ],
    },
)