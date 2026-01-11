
# display help information
default:
    @just --list


# clean the project
clean:
    rm -rf .venv/


# install the dependencies
install:
    uv sync --all-groups


# format with black and ruff
format:
    uv run ruff format

precommit:
	uv run pre-commit run --all-files