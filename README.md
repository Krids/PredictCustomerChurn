# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

The Project is aimed at identifying credit card customers who are most likely to churn. Machine learning algorithms have been used to achieve this objective. PEP8 standard has been followed while writing the Python code. Python Codes could be found in the repository along with the testing and logging python script.

## Requirements

1. Let poetry create the virtual environment for you.

```bash
poetry install
```

2. Create the virtual environment beforehand and then poetry will follow your lead.

```bash
python -m venv .venv && \
  source .venv/bin/activate &&
  poetry install
```

### Dependencies

- python = "^3.8"
- pandas = "^1.4.0"
- numpy = "^1.22.2"
- matplotlib = "^3.5.1"
- notebook = "^6.4.8"
- jupyter = "^1.0.0"
- autopep8 = "^1.6.0"
- pylint = "^2.12.2"
- shap = "^0.40.0"
- seaborn = "^0.11.2"
- pytest = "^7.0.0"

When your environment is ready, you can execute this project.

```bash
python main.py
```

To run the tests of this project you should run the following command on root folder.

```bash
python -m unittest tests.test_churn_library
```

Then all the tests should run and produce the logs in the docs/logs folder.
