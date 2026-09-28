#!/usr/bin/env bash
set -euo pipefail
conda create -n hou48 python=3.11 -y
conda run -n hou48 conda install -c conda-forge rdkit biopython pandas numpy scipy scikit-learn matplotlib jupyter tqdm -y
conda run -n hou48 pip install torch
