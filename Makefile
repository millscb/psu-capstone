exclude=.venv,htmrl_env,.pytest_cache,notebooks,reports,

.PHONY: help install format lint clean test update setup-dev setup-uv-windows setup-uv pre-commit env-setup recreate-venv

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Quick start: 'make install'"

ifeq ($(OS),Windows_NT)
install: ## Install package and pre-commit hooks (Windows)
	@echo "📦 Installing package in editable mode..."
	@IF EXIST ".venv" ( \
		echo "ℹ️ Existing .venv detected. Skipping Python reinstall."; \
	) ELSE ( \
		echo "🚀 Creating new uv environment..."; \
		uv python install 3.12 && uv python pin 3.12; \
	)
	@uv lock --upgrade
	@uv sync --all-groups
	@git rev-parse --git-dir >nul 2>&1 || (echo "⚠️ Git repository not initialized. Initializing..." && git init && git branch -m main && echo "✅ Git repository initialized with main branch")
	@echo "🔧 Setting up pre-commit hooks..."
	@uv run --active pre-commit install
	@echo "✅ Installation complete"
else
install: ## Install package and pre-commit hooks (Unix)
	@echo "📦 Installing package in editable mode..."
	@if [ -d ".venv" ]; then \
		echo "ℹ️ Existing .venv detected. Skipping Python reinstall."; \
	else \
		echo "🚀 Creating new uv environment..."; \
		uv python install 3.12 && uv python pin 3.12; \
	fi
	@uv lock --upgrade
	@uv sync --all-groups
	@git rev-parse --git-dir >/dev/null 2>&1 || (echo "⚠️ Git repository not initialized. Initializing..." && git init && git branch -m main && echo "✅ Git repository initialized with main branch")
	@echo "🔧 Setting up pre-commit hooks..."
	@uv run --active pre-commit install
	@echo "✅ Installation complete"
endif

recreate-venv: ## Force recreate uv virtual environment
	@echo "🧪 Recreating uv virtual environment..."
ifeq ($(OS),Windows_NT)
	@IF EXIST ".venv" (rmdir /S /Q .venv) || echo "ℹ️ No existing .venv"
else
	@rm -rf .venv || true
endif
	@uv python install 3.12
	@uv python pin 3.12
	@uv sync --all-groups
	@echo "✅ Environment recreated"

setup-dev: ## Setup development environment
	@echo "📚 Installing development dependencies..."
	@uv sync --all-groups
	@echo "✅ Development environment ready. Try `make test` to verify everything works"

format: ## Format code with isort and black
	@echo "🎨 Formatting code..."
	@uv run --active --no-sync isort . --line-length=100
	@uv run --active --no-sync black . --line-length=100
	@echo "✅ Code formatted"

lint: ## Run linting checks
	@echo "🔍 Running linting checks..."
	@uv run --isolated --with flake8 --with flake8-pyproject --with pep8-naming python -m flake8 . --config=.flake8 --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=$(exclude) -v
	@uv run --isolated --with flake8 --with flake8-pyproject --with pep8-naming python -m flake8 . --config=.flake8 --count --show-source --max-complexity=10 --statistics --exclude=$(exclude)
	@echo "✅ Linting complete"

lint-docs: ## Check docstring coverage and style
	@echo "📝 Checking docstring coverage and style..."
	@uv run --active --no-sync pydocstyle src/htmrl src/utils.py --convention=google --add-ignore=D100,D104,D105,D107 || echo "⚠️ Found docstring style issues"
	@uv run --active --no-sync interrogate -vv src/htmrl src/htmrl/utils.py src/htmrl/grapher.py --fail-under=80 --ignore-init-method --ignore-magic --exclude tests
	@echo "✅ Docstring checks complete"

lint-docs-strict: ## Strict docstring validation with pydoclint
	@echo "📝 Running strict docstring validation..."
	@uv run --with pydoclint --with docstring-parser-fork pydoclint --style=google --exclude='\.venv|tests|build' src/
	@echo "✅ Strict docstring validation complete"

clean:
	@echo "🧹 Cleaning build artifacts..."
ifeq ($(OS),Windows_NT)
	@IF EXIST "target" (rmdir /S /Q target) || echo "ℹ️ no target"
	@IF EXIST "dist" (rmdir /S /Q dist) || echo "ℹ️ no dist"
	@IF EXIST "build" (rmdir /S /Q build) || echo "ℹ️ no build"
	@for /d %%D in (*.egg-info) do @rmdir /S /Q "%%D"
	@IF EXIST "htmlcov" (rmdir /S /Q htmlcov) || echo "ℹ️ no htmlcov"
	@IF EXIST ".coverage" (del /Q /F .coverage) || echo "ℹ️ no .coverage"
	@IF EXIST ".pytest_cache" (rmdir /S /Q .pytest_cache) || echo "ℹ️ no .pytest_cache"
	@for /r %%D in (__pycache__) do @rmdir /S /Q "%%D"
else
	@rm -rf target/* dist/* build/* *.egg-info htmlcov .coverage .pytest_cache
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
endif
	@echo "✅ Cleanup complete"

update: ## Update dependencies
	@echo "🔺 Updating dependencies..."
	@uv lock --upgrade
	@echo "✅ Dependencies updated"

## In order to run a specific test file or directory, use:
## make test ARGS="-v tests/test_file_or_directory"
test: ## Run tests with coverage
	@echo "🧪 Running tests with coverage..."
ifeq ($(OS),Windows_NT)
	@IF NOT DEFINED ARGS ( \
		set PYTHONPATH=src && uv run pytest ^ \
			--cov="psu_capstone" ^ \
			--cov-report=term-missing ^ \
			--cov-report=html:htmlcov ^ \
			--durations=0 ^ \
			--disable-warnings ^ \
			tests/ \
	) ELSE ( \
		set PYTHONPATH=src && uv run pytest ^ \
			--cov="psu_capstone" ^ \
			--cov-report=term-missing ^ \
			--cov-report=html:htmlcov ^ \
			--durations=0 ^ \
			--disable-warnings ^ \
			$(ARGS) \
	)
else
	@if [ -z "$(ARGS)" ]; then \
			PYTHONPATH=src/ uv run --active pytest \
					--cov="psu_capstone" \
					--cov-report=term-missing \
					--cov-report=html:htmlcov \
					--durations=0 \
					--disable-warnings \
					tests/; \
	else \
			PYTHONPATH=src/ uv run --active pytest \
					--cov="psu_capstone" \
					--cov-report=term-missing \
					--cov-report=html:htmlcov \
					--durations=0 \
					--disable-warnings \
					$(ARGS); \
	fi
endif
	@echo "✅ Tests complete. Coverage report: htmlcov/index.html"

setup-uv-windows: ## Install uv package manager on Windows
	@echo "🚀 Installing uv package manager..."
	@powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
	@echo "✅ uv installed successfully"

setup-uv: ## Install uv package manager on Unix systems (Linux/MacOS)
	@echo "🚀 Installing uv package manager..."
	@curl -LsSf https://astral.sh/uv/install.sh | sh
	@echo "✅ uv installed successfully"

pre-commit: ## Run pre-commit on all files
	@echo "🔧 Running pre-commit on all files..."
	@uv run --active pre-commit run --all-files
