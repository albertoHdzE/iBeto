#!/usr/bin/env bash

set -e

echo "Creating iBeto project structure..."

mkdir -p \
ibeto/{audio,cli,config,core,llm,memory,prompts,ui,utils} \
docs/{architecture,decisions,proposals,research} \
tests/{unit,integration} \
experiments/backend_benchmark/results \
scripts \
configs \
models \
logs \
assets/{audio,images}

touch \
ibeto/audio/__init__.py \
ibeto/cli/__init__.py \
ibeto/config/__init__.py \
ibeto/core/__init__.py \
ibeto/llm/__init__.py \
ibeto/memory/__init__.py \
ibeto/prompts/__init__.py \
ibeto/ui/__init__.py \
ibeto/utils/__init__.py

echo "Done!"