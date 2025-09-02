# Advanced Data Science Tutorial: World Happiness Report Analysis

This repository demonstrates advanced skills in data analysis, data science, software engineering, and math using the World Happiness Report dataset (2015-2024). It is structured for JOSS submission: modular code, tests, docs, and reproducibility.

## Installation
1. Clone the repo: `git clone https://github.com/yourusername/advanced-data-science-tutorial.git`
2. Download datasets from Kaggle:
   - Up to 2019: https://www.kaggle.com/datasets/unsdsn/world-happiness
   - 2020-2024: https://www.kaggle.com/datasets/ajaypalsinghlo/world-happiness-report-2024
   - 2025: https://www.kaggle.com/datasets/rmarbun/world-happiness-report-2025
   Place all CSVs (e.g., 2015.csv, ..., 2025.csv) in `data/`.
3. Set up virtual environment:

```
python -m venv .venv && source .venv/bin/activate
```

4. Install the package editable with dev tools: `pip install -e .[dev]`

## Usage
- Run the tutorial: Open `notebooks/advanced_tutorial.ipynb` in Jupyter.
- CLI example: `python -m world_happiness.models` (after installation).
- Build docs: `cd docs && make html`
- Tests: `pytest`
- Docker: `docker build -t happiness-tutorial . && docker run happiness-tutorial`

## Citation
See CITATION.cff.

## License
MIT