#!/usr/bin/env bash

# NVM may be initialized before Conda and otherwise shadows this environment's Node.
export RL4PPT_FINAL_PATH_BEFORE_ACTIVATE="$PATH"
export PATH="$CONDA_PREFIX/bin:$PATH"
