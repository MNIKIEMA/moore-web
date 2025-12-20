
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
    uv tool run black src
    uv tool run ruff format