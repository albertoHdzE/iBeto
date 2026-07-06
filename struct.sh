#!/usr/bin/env bash

set -e

echo "Creating iBeto project structure..."

mkdir -p \
docs/proposals \
docs/architecture \
docs/decisions \
docs/research \
ibeto/core \
ibeto/audio \
ibeto/llm \
ibeto/memory \
ibeto/prompts \
ibeto/config \
ibeto/utils \
ibeto/ui \
scripts \
tests \
experiments \
models \
configs \
logs \
assets/audio \
assets/images

touch \
README.md \
LICENSE \
.gitignore \
pyproject.toml \
ibeto/__init__.py

cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*.so

# Virtual environments
.venv/
.env

# IDE
.vscode/
.idea/

# macOS
.DS_Store

# Logs
logs/

# Python tooling
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Models
models/**/*.gguf
models/**/*.bin
models/**/*.safetensors

# LM Studio cache
lmstudio/

# uv
uv.lock

# Misc
*.log
EOF

echo "Done."