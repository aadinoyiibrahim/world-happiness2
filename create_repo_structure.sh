#!/bin/bash

# Create the root directory
cd world-happiness

# Create root-level files
touch README.md
touch paper.md
touch CITATION.cff
touch LICENSE
touch pyproject.toml
touch requirements.txt

# Create data directory (empty, as noted)
mkdir -p data

# Create src subdirectory and files
mkdir -p src/world_happiness
touch src/world_happiness/__init__.py
touch src/world_happiness/dataloader.py
touch src/world_happiness/eda.py
touch src/world_happiness/models.py
touch src/world_happiness/math_utils.py

# Create notebooks subdirectory and file
mkdir -p notebooks
touch notebooks/advanced_tutorial.ipynb

# Create tests subdirectory and files
mkdir -p tests
touch tests/test_dataloader.py
touch tests/test_eda.py
touch tests/test_models.py
touch tests/test_math_utils.py

# Create docs subdirectory and files
mkdir -p docs
touch docs/conf.py
touch docs/index.rst
touch docs/requirements.txt

# Create Dockerfile
touch Dockerfile

# Create .github/workflows subdirectory and file
mkdir -p .github/workflows
touch .github/workflows/ci.yml

echo "Repository structure created successfully!"