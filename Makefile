.PHONY: entrypoint install train deploy start_port_forward stop_port_forward build-test-image run-test-job"

ENVIRONMENT ?= dev-

entrypoint: deploy run-test-job train

make refresh:
	@echo "Refreshing resources..."
	@helm uninstall training-infra
	@helm upgrade --install training-infra ./training-infra --set environment=${ENVIRONMENT}

deploy:
	@echo "Deploying resources..."
	@helm upgrade --install training-infra ./training-infra --set environment=${ENVIRONMENT}
	@echo "Waiting for jobs to complete..."
	@bash ./scripts/wait_for_jobs.sh

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

run-test-job:
	@echo "Running test job..."
	# make sure the proper env is used in the tester.yaml - this should be done with
	# proper template, but do it like this to quickly check if it works
	@cat tester.yaml | sed "s/ENV/${ENVIRONMENT}/g" > tester-tmp.yaml && mv tester-tmp.yaml tester.yaml
	@kubectl delete job tests --ignore-not-found
	@env | grep -v PATH > local.env
	@kubectl create configmap test-env --from-env-file=local.env --dry-run=client -o yaml | kubectl apply -f -
	@kubectl apply -f tester.yaml
	@echo "Waiting for test job to complete..."
	@kubectl wait --for=condition=complete job/tests --timeout=300s || kubectl wait --for=condition=failed job/tests --timeout=300s
	@echo "Test job logs:"
	@kubectl logs job/tests
