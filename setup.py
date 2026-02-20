#!/usr/bin/env python3
"""Minimal setup.py for editable install of GX1_ENGINE"""
from setuptools import setup, find_packages

setup(
    name="gx1",
    version="0.1.0",
    packages=find_packages(),
    package_dir={"": "."},
)
