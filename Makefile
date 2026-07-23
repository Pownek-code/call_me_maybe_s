.PHONY: install run debug clean lint lint-strict

# Directories excluded from linting: the virtualenv (third-party code) and the
# provided SDK (copied in per the subject, not ours to modify).
LINT_EXCLUDE = .venv,llm_sdk,__pycache__,.mypy_cache
MYPY_FLAGS = --warn-return-any --warn-unused-ignores --ignore-missing-imports \
             --disallow-untyped-defs --check-untyped-defs

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache

lint:
	uv run flake8 . --exclude=$(LINT_EXCLUDE)
	uv run mypy src $(MYPY_FLAGS)


lint-strict:
	uv run flake8 . --exclude=$(LINT_EXCLUDE)
	uv run mypy src --strict
