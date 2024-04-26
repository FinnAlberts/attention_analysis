.ONESHELL:
SHELL=/bin/bash

help: ## Show this help
	@egrep '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

test: ## Run pytest and report coverage
	pytest --cov-report term-missing --cov=src

clean: ## Remove build files
	rm -Rf ./dist/

install-poetry: ## Install poetry for Linux
	curl -sSL https://install.python-poetry.org | python3 -

.PHONY: help init test
