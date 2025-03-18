.PHONY: entrypoint install train deploy start_port_forward stop_port_forward

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

start_port_forward:
	@echo "Starting port forwarding..."
	@kubectl port-forward service/raycluster-${ENVIRONMENT}-head-svc 8265:8265 10001:10001 & echo $$! >> .port_forward.pid

stop_port_forward:
	@echo "Stopping port forwarding..."
	@if [ -f .port_forward.pid ]; then \
		kill `cat .port_forward.pid`; rm .port_forward.pid; \
	else \
		echo "No port forwarding process found"; \
	fi

install:
	uv sync

train: install start_port_forward
	@trap 'make stop_port_forward' EXIT 2 15; \
	echo "Training model..." && \
	RAY_AIR_NEW_OUTPUT=0 uv run --active python training/train.py
