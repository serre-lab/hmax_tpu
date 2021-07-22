#!/usr/bin/env bash

os mkdir .bspipeline
cd .bspipeline 
git clone https://github.com/serre-lab/brain-score .bspipeline/
git clone https://github.com/serre-lab/candidate-models .bspipeline/
pip install -e .bspipeline/candidate-models/
pip install -e  .bspipeline/brain-score/

