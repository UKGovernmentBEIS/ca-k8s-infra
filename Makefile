.PHONY: entrypoint install train deploy start_port_forward stop_port_forward

entrypoint: deploy train

make refresh:
	@echo "Refreshing resources..."
	@kubectl delete -f infra/manifests/
	@kubectl apply -f infra/manifests/

deploy:
	@echo "Deploying resources..."
	@kubectl apply -f infra/manifests/
	@echo "Waiting for jobs to complete..."
	@kubectl get jobs -o name | xargs -I {} bash -c 'kubectl wait --for=condition=complete {} --timeout=300s || kubectl wait --for=condition=failed {} --timeout=300s'


start_port_forward:
	@echo "Starting port forwarding..."
	@kubectl port-forward --namespace training service/raycluster-head-svc 8265:8265 10001:10001 & echo $$! >> .port_forward.pid
	@kubectl port-forward --namespace kube-system service/hubble-ui 8080:80 & echo $$! >> .port_forward.pid

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
