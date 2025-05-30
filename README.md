# Training Setup

This repository contains Infrastructure as Code (IaC) and code definitions for training ML models using Ray on Kubernetes. The setup includes distributed training capabilities and secure access to model weights and datasets through MinIO.

## Prerequisites

- Access to the Kubernetes cluster
- `kubectl` configured with the correct context
- `uv` package manager installed

## Infrastructure

The training infrastructure consists of:

- A Ray cluster with 1 head node and 2 worker nodes
- Cilium Network Policies for secure communication
- Integration with internal MinIO for model weights and datasets storage

## Usage

### Deployment

Deploy the Ray cluster and network policies:

```bash
make deploy
```

To refresh the deployment (delete and redeploy):

```bash
make refresh
```

You should run this if the ray worker image has been updated.

### Training

Start a training job:

```bash
make train
```

This command will:

- Install dependencies
- Set up port forwarding to the Ray cluster
- Run the training script
- Automatically clean up port forwarding on completion
- Port Forwarding
- Manually manage port forwarding:

### Start port forwarding

```bash
make start_port_forward
```

This will make the Ray dashboard accessible on `localhost:8265` and the Hubble dashboard accessible on `localhost:8080`.

### Stop port forwarding

```bash
make stop_port_forward
```

## Storage

Model weights and datasets are stored in the internal MinIO deployment, accessible at minio.minio-internal.svc.cluster.local. The training setup includes utilities for copying data between S3 and local storage:

```python
fs = pyarrow.fs.S3FileSystem(
    access_key="minioadmin",
    secret_key="minioadmin",
    endpoint_override="minio.minio-internal.svc.cluster.local:9000",
    scheme="http"
)
```

## Docker Images

All dockerfiles in the `training-infra/dockerfiles` repo are built externally and pushed with the name of the dockerfile. Any updates to these files in a PR will automatically cause the images to be built and pushed. They can then be accessed with e.g. `localhost:5001/<dockerfile_name>:{{ .Values.environment }}` by helm templates.

## Developer Notes

The training script automatically handles data and model transfer between MinIO and local storage.
