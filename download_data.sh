#!/usr/bin/env bash
set -euo pipefail
mkdir -p data
cd data
curl -L "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/pdbbind_v2013_core_set.tar.gz" -o pdbbind_v2013_core_set.tar.gz
tar -xzf pdbbind_v2013_core_set.tar.gz
