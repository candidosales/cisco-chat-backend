# provide ENV=dev to use .env.dev instead of .env
# and to work in the Pulumi dev stack
ENV_LOADED :=

ifeq ($(ENV), prod)
    ifneq (,$(wildcard ./.env))
        include .env
        export
				ENV_LOADED := Loaded config from .env
    endif
else
    ifneq (,$(wildcard ./.env.dev))
        include .env.dev
        export
				ENV_LOADED := Loaded config from .env.dev
    endif
endif

check-env-variables:  ## pushes secrets from .env to Modal
	@$(if $(value OPENAI_API_KEY),, \
		$(error OPENAI_API_KEY is not set. Please set it before running this target.))

document-store: check-env-variables ## Read JSON and store in Chroma DB
	@tasks/pretty_log.sh "See docstore.py and the ETL notebook for details"
	tasks/run_etl.sh --drop --db $(MONGODB_DATABASE) --collection $(MONGODB_COLLECTION)


environment: ## installs required environment for deployment and corpus generation
	@if [ -z "$(ENV_LOADED)" ]; then \
			echo "Error: Configuration file not found" >&2; \
			exit 1; \
    else \
			tasks/pretty_log.sh "$(ENV_LOADED)"; \
	fi
	python -m pip install -qqq -r requirements.txt

dev-environment: environment  ## installs required environment for development
	python -m pip install -qqq -r requirements-dev.txt