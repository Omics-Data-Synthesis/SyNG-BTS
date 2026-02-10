# SyNG-BTS Makefile
# Development automation for SyNG-BTS package

.PHONY: help install install-dev install-docs install-all build publish publish-test docs docs-serve clean clean-all lint format test test-cov init-dev check

# Python executable - uses venv if available, otherwise system Python
PYTHON := $(shell if [ -f .venv/bin/python ]; then echo ".venv/bin/python"; else echo "python"; fi)
PIP := $(shell if [ -f .venv/bin/pip ]; then echo ".venv/bin/pip"; else echo "pip"; fi)

# Default target
help:
	@echo "SyNG-BTS Development Commands"
	@echo "=============================="
	@echo ""
	@echo "Installation:"
	@echo "  make install       - Install the package"
	@echo "  make install-dev   - Install with development dependencies"
	@echo "  make install-docs  - Install with documentation dependencies"
	@echo "  make install-all   - Install with all optional dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make init-dev      - Initialize development environment (venv + deps)"
	@echo "  make test          - Run tests"
	@echo "  make test-cov      - Run tests with coverage report"
	@echo "  make check         - Run all checks (lint, test)"
	@echo "  make lint          - Run linters (ruff)"
	@echo "  make format        - Format code (ruff format)"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs          - Build documentation"
	@echo "  make docs-serve    - Build and serve docs locally"
	@echo ""
	@echo "Distribution:"
	@echo "  make build         - Build distribution packages"
	@echo "  make publish       - Publish to PyPI"
	@echo "  make publish-test  - Publish to TestPyPI"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         - Remove build artifacts"
	@echo "  make clean-all     - Remove all generated files"

# =============================================================================
# Installation
# =============================================================================

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

install-docs:
	$(PIP) install -e ".[docs]"

install-all:
	$(PIP) install -e ".[all]"

init-dev:
	@echo "Creating virtual environment..."
	python -m venv .venv
	@echo "Installing package with dev dependencies..."
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e ".[dev]"
	@echo ""
	@echo "Development environment ready!"
	@echo "Activate with: source .venv/bin/activate"

# =============================================================================
# Development
# =============================================================================

lint:
	$(PYTHON) -m ruff check syng_bts/ tests/

format:
	$(PYTHON) -m ruff format syng_bts/ tests/
	$(PYTHON) -m ruff check --fix syng_bts/ tests/

test:
	$(PYTHON) -m pytest tests/ -v

test-cov:
	$(PYTHON) -m pytest tests/ -v --cov=syng_bts --cov-report=html --cov-report=term
	@echo ""
	@echo "Coverage report generated: htmlcov/index.html"

check: lint test
	@echo "All checks passed!"

# =============================================================================
# Documentation
# =============================================================================

docs:
	cd docs && make html
	@echo ""
	@echo "Documentation built: docs/build/html/index.html"

docs-serve: docs
	@echo "Starting local documentation server..."
	$(PYTHON) -m http.server 8000 --directory docs/build/html

# =============================================================================
# Build & Publish
# =============================================================================

build: clean
	$(PYTHON) -m build
	@echo ""
	@echo "Built packages:"
	@ls -la dist/

publish: build
	@printf "Are you sure you want to publish to PyPI? Type 'yes' to continue (default: no): "; \
	read -r resp; \
	resp=$$(echo "$$resp" | tr '[:upper:]' '[:lower:]'); \
	if [ "$$resp" != "yes" ]; then \
		echo "Publish aborted."; \
		exit 1; \
	fi; \
	@echo "Publishing to PyPI..."
	$(PYTHON) -m twine upload dist/*

publish-test: build
	@printf "Are you sure you want to publish to TestPyPI? Type 'yes' to continue (default: no): "; \
	read -r resp; \
	resp=$$(echo "$$resp" | tr '[:upper:]' '[:lower:]'); \
	if [ "$$resp" != "yes" ]; then \
		echo "Publish aborted."; \
		exit 1; \
	fi; \
	@echo "Publishing to TestPyPI..."
	$(PYTHON) -m twine upload --repository testpypi dist/*

# =============================================================================
# Cleanup
# =============================================================================

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf syng_bts/*.egg-info/
	rm -rf .eggs/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

clean-all: clean
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf docs/build/
	# Clean generated output directories
	rm -rf GeneratedData/
	rm -rf Loss/
	rm -rf ReconsData/
	rm -rf Transfer/
