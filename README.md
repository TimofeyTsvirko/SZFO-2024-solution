# Project README! Here we go!

<div align="center">

[![PythonSupported](https://img.shields.io/badge/python-3.10-brightgreen.svg)](https://python3statement.org/#sections50-why)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)

Awesome `szfo-2024-solution` project!

</div>

- [Repository contents](#repository-contents)
- [System requirements](#system-requirements)
- [Other interesting info](#other-interesting-info)

## Repository contents

- [docs](docs) - documentation of the project
- [reports](reports) - reports generated (as generated from notebooks)
  > Check if you need to ignore large reports or keep them in Git LFS
- [configs](configs) - configuration files directory

- [notebooks](notebooks) - directory for `jupyter` notebooks

- [scripts](scripts) - repository service scripts
  > These ones are not included into the pakckage if you build one - these scripts are only for usage with repository
- [szfo_2024_solution](szfo_2024_solution) - source files of the project
- [.editorconfig](.editorconfig) - configuration for [editorconfig](https://editorconfig.org/)
- [.flake8](.flake8) - [flake8](https://github.com/pycqa/flake8) linter configuration
- [.gitignore](.gitignore) - the files/folders `git` should ignore
- [.pre-commit-config.yaml](.pre-commit-config.yaml) - [pre-commit](https://pre-commit.com/) configuration file
- [README.md](README.md) - the one you read =)
- [DEVELOPMENT.md](DEVELOPMENT.md) - guide for development team
- [Makefile](Makefile) - targets for `make` command
- [cookiecutter-config-file.yml](cookiecutter-config-file.yml) - cookiecutter project config log
- [poetry.toml](poetry.toml) - poetry local config
- [pyproject.toml](pyproject.toml) - Python project configuration

## System requirements

- Python version: 3.10
- Operating system: Ubuntu or WSL
- Poetry version >= 1.2.0

> We tested on this setup - you can try other versions or operation systems by yourself!

## Other interesting info

Here you can write anything about your project!

# Get submission

To get submission results you need to build the project (DEVELOPMENT.md) and run:
```bash
python scripts/get_submission.py --src ... --dst ...
```

- If you have the following issue:
```bash
Traceback (most recent call last):
  File "/.../SZFO-2024-solution/scripts/process_folder.py", line 13, in <module>
    from szfo_2024_solution.voice2text import MetricsCalculator, VoskASR
ModuleNotFoundError: No module named 'szfo_2024_solution'
```

Run in the terminal:
```bash
export PYTHONPATH="/your_dir_here/SZFO-2024-solution:$PYTHONPATH"
```

And run the script again
