# Development Guide

To contribute to this repository, please follow theses steps:

* Fork the repository at `https://github.com/VsonicV/es-fine-tuning-paper` by clicking on the `Fork` button

* Clone your fork. E.g., if your GitHub ID is `abc123`, run the following command in a terminal:

    ```bash
    git clone https://github.com/abc123/es-fine-tuning-paper
    ```

* Create a feature branch off the `main` branch of your fork. E.g., for a feature branch named `peft`,
go to `es-fine-tuning-paper` folder and run the following command in a terminal:

    ```bash
    git checkout -b peft
    ```

* Commit your changes to your feature branch in your fork

* To ensure your code is free of lint errors and is formatted correctly, do the following:

    1. Create a virtual environment with python version >= 3.10 (if you have not done already),
    activate it, and set the PYTHONPATH environment variable:

        ```bash
        python -m venv es
        source es/bin/activate
        export PYTHONPATH=`pwd`
        ```

    2. Install the build requirements:

        ```bash
        pip install -r requirements-build.txt
        ```

    3. Run the following commnds to format your code using isort and black:

        ```bash
        ./es/bin/isort isort --skip es .
        ./es/bin/black --exclude 'es/.*' .
        ```

    4. Run the following commands to catch lint errors using flake8 and pylint:

        ```bash
        ./es/bin/flake8 --exclude=es .
        ./es/bin/pylint pylint --recursive=y . --ignore=es
        ```

    5. Run the following command to make sure your markdown file is formatted correctly:

        ```bash
        ./es/bin/pymarkdown --config ./.pymarkdownlint.yaml scan ./docs/**/*.md ./README.md
        ```

* Create a Pull Request (PR) in `https://github.com/VsonicV/es-fine-tuning-paper` by clicking on the
`Pull Requests` button, then clicking on `New pull request` buton. In the `Compare changes` pagem click
on `compare across forks`, and select your fork along with your feature branch.

* Revise your code per reviewer comments. Make sure your changes do not introduce any lint or formatting errors
