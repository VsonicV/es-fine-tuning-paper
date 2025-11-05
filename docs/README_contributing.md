# Development Guide

Create a virtual environment with python version >= 3.10 (if you have not
done already), activate it, and set the PYTHONPATH environment variable:

```bash
python -m venv es
source es/bin/activate
export PYTHONPATH=`pwd`
```

Install the build requirements:

```bash
pip install -r requirements-build.txt
```

Run the following commnds to format your code using isort and black:

```bash
./es/bin/isort es_fine-tuning_conciseness.py
./es/bin/black es_fine-tuning_conciseness.py
```

Run the following commands to catch lint errors using flake8 and pylint:

```bash
./es/bin/flake8 es_fine-tuning_conciseness.py
./es/bin/pylint es_fine-tuning_conciseness.py
```

Run the following command to make sure your markdown file is formatted correctly:

```bash
./venv/bin/pymarkdown --config ./.pymarkdownlint.yaml scan ./docs/**/*.md ./README.md
```
