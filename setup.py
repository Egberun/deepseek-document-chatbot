"""
Setup script for DeepSeek Document Chatbot package
"""

from setuptools import setup, find_packages
import os
import re

# Read version from __init__.py
with open("__init__.py", "r") as f:
    version_match = re.search(r"__version__\s*=\s*['\"]([^'\"]*)['\"]", f.read())
    version = version_match.group(1) if version_match else "0.0.1"

# Read requirements from requirements.txt
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

# Read README.md for long description
with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="deepseek-document-chatbot",
    version=version,
    description="A document-based chatbot system using DeepSeek LLM for intelligent question answering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Egberun",
    author_email="",
    url="https://github.com/Egberun/deepseek-document-chatbot",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "deepseek-chatbot=main:main",
        ],
    },
)