.PHONY: entrypoint install train deploy

ENVIRONMENT ?= dev-

entrypoint: deploy train

make refresh:
	@echo "Refreshing resources..."
	@helm uninstall training-infra 
	@helm upgrade --install training-infra ./training-infra --set environment=${ENVIRONMENT}

deploy:
	@echo "Deploying resources..."
	@helm upgrade --install training-infra  ./training-infra --set environment=${ENVIRONMENT}
	@echo "Waiting for jobs to start..."
	@kubectl get jobs -o name | xargs -I {} bash -c 'kubectl wait --for=jsonpath="{.status.ready}=1" {} --timeout=30s || true'
	@echo "Waiting for jobs to complete..."
	@kubectl get jobs -o name | xargs -I {} bash -c 'kubectl logs -f {} || true'

install:
	uv sync

train: install
	echo "Training model..." && \
	RAY_AIR_NEW_OUTPUT=0 uv run --active python training/train.py
