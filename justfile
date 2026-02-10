
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

# perform pre-commit checks with prek
precommit:
	uv run prek run

# run type checking with ty
typecheck:
    uvx ty check
