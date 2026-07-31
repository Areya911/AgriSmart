#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status
set -e

echo "===> Installing build dependencies with pinned setuptools < 70.0.0..."
pip install "setuptools<70.0.0" wheel

echo "===> Installing openai-whisper with --no-build-isolation..."
pip install --no-build-isolation openai-whisper

echo "===> Installing remaining requirements..."
pip install -r requirements.txt

echo "===> Build completed successfully!"
