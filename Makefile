.PHONY: entrypoint install train deploy start_port_forward stop_port_forward refresh

ENVIRONMENT ?= dev-

entrypoint: refresh train

deploy:
	@echo "Removing unfinished jobs..."
	@kubectl get jobs -o json | jq -r '.items[] | select((.status.failed // 0) >= 1 or ((.status.active // 0) >= 1 and (.status.succeeded // 0) == 0)) | .metadata.name' | xargs -r kubectl delete job --wait=true
	@echo "Deploying resources..."
	@helm upgrade --install training-infra ./training-infra --set environment=${ENVIRONMENT}
	@echo "Waiting for jobs to complete..."
	@bash ./scripts/wait_for_jobs.sh

make refresh:
	@if helm status training-infra >/dev/null 2>&1; then \
		helm uninstall training-infra; \
	fi
	@$(MAKE) deploy

wait_for_ray:
	@echo "Waiting for Ray cluster head node to be ready..."
	@RETRY=0; DELAY=2; \
	while [ $$RETRY -lt 6 ]; do \
		POD_NAME=$$(kubectl get endpoints raycluster-${ENVIRONMENT}-head-svc -o jsonpath='{.subsets[0].addresses[0].targetRef.name}' 2>/dev/null); \
		echo "Checking for pod: $$POD_NAME"; \
		if [ -n "$$POD_NAME" ]; then \
			echo "Found pod: $$POD_NAME"; \
			kubectl wait --for=condition=ready "pod/$$POD_NAME" --timeout=60s; \
			break; \
		else \
			RETRY=$$((RETRY + 1)); \
			echo "Attempt $$RETRY/5: Pod not found, waiting $$DELAY seconds..."; \
			sleep $$DELAY; \
			DELAY=$$((DELAY * 2)); \
		fi; \
	done; \
	if [ $$RETRY -eq 5 ]; then \
		echo "Error: Could not find pod after 5 attempts"; \
		exit 1; \
	fi

start_port_forward: wait_for_ray
	@echo "Starting port forwarding..."
	@kubectl port-forward service/raycluster-${ENVIRONMENT}-head-svc 8265:8265 10001:10001 & echo $$! >> .port_forward.pid

stop_port_forward:
	@echo "Stopping port forwarding..."
	@if [ -f .port_forward.pid ]; then \
		kill `cat .port_forward.pid`; rm .port_forward.pid; \
	else \
		echo "No port forwarding process found"; \
	fi

train: start_port_forward
	@trap 'make stop_port_forward' EXIT 2 15; \
	echo "Training model..." && \
	RAY_AIR_NEW_OUTPUT=0 uv run --active python training/train.py
