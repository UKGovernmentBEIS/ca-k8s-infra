import os
import sys
import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory
import logging
import logging.handlers

import pyarrow
import ray
import torch
from datasets import load_from_disk
from elasticsearch import Elasticsearch
from pyarrow.fs import S3FileSystem
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer, prepare_data_loader, prepare_model
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


runtime_env = {
    "env_vars": {
        "ENVIRONMENT": os.environ["ENVIRONMENT"],
    }
}

# Define the scaling configuration
scaling_config = ScalingConfig(
    num_workers=2,
    use_gpu=False,
    resources_per_worker={"CPU": 1, "memory": 4 * 1024 * 1024 * 1024},
)

# Training configuration
train_config = {
    "model_location": "model-small",  # This is the path in MinIO under the models bucket
    "batch_size": 2,
    "learning_rate": 5e-5,
    "num_epochs": 1,
    "max_iterations": 10,
}

def setup_worker_logging():
    """Configure logging for the Ray worker to write to stdout"""
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid duplicate logging
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add a stream handler to write to stdout in a structured format
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter('[TRAINING] %(message)s'))
    root_logger.addHandler(stream_handler)

    # Configure Ray loggers
    ray_loggers = [
        "ray.data",
        "ray.tune",
        "ray.rllib",
        "ray.train",
        "ray.serve"
    ]

    for logger_name in ray_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        # Don't add handler - it will inherit from root logger

    # Log initial worker information
    logging.info(f"Python version: {sys.version}")
    logging.info(f"Working directory: {os.getcwd()}")
    logging.info("Logging configured to write to stdout in JSON format")

# For copying from S3 to local:
def copy_dir_from_s3(fs, source_path, target_path):
    file_info = fs.get_file_info(pyarrow.fs.FileSelector(source_path))
    for file in file_info:
        if file.type == pyarrow.fs.FileType.Directory:
            subdir = os.path.basename(file.path)
            os.makedirs(os.path.join(target_path, subdir), exist_ok=True)
            copy_dir_from_s3(fs, file.path, os.path.join(target_path, subdir))
        else:
            os.makedirs(target_path, exist_ok=True)
            with (
                fs.open_input_file(file.path) as s3_file,
                open(
                    os.path.join(target_path, os.path.basename(file.path)), "wb"
                ) as local_file,
            ):
                local_file.write(s3_file.read())


# For copying from local to S3:
def copy_dir_to_s3(fs, source_path, target_path):
    for root, _, files in os.walk(source_path):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, source_path)
            s3_path = os.path.join(target_path, relative_path)
            with (
                open(local_path, "rb") as local_file,
                fs.open_output_stream(s3_path) as s3_file,
            ):
                s3_file.write(local_file.read())


def train_func(config):
    # Get the current worker's rank
    rank = ray.train.get_context().get_world_rank()
    
    # Set up logging for this worker
    setup_worker_logging()
    logging.info(f"Starting training function in worker {rank}")

    es = Elasticsearch(
        [
            f"http://elasticsearch.observability-{os.environ['ENVIRONMENT']}.svc.cluster.local:9200"
        ]
    )

    es.index(index="logs", document={"message": "Started training model"})

    # Initialize the S3 filesystem
    fs = S3FileSystem(
        access_key="minioadmin",
        secret_key="minioadmin",  # noqa
        endpoint_override=f"minio.minio-internal-{os.environ['ENVIRONMENT']}.svc.cluster.local:9000",
        scheme="http",
    )

    # copy data to disk
    logging.info("Copying data to disk")
    with TemporaryDirectory() as tmp_dir:
        imdb_path = Path(tmp_dir)
        copy_dir_from_s3(fs, "datasets/imdb/", imdb_path)
        logging.info("Data copied to disk")

        # Load datasets
        train_dataset = load_from_disk(str(imdb_path / "train"))
        test_dataset = load_from_disk(str(imdb_path / "test"))

    # Take subset of data
    train_dataset = train_dataset.take(train_config["batch_size"])
    test_dataset = test_dataset.take(train_config["batch_size"])

    # Prepare local directory for the model files
    s3_path = f"models/{config['model_location']}/"

    with tempfile.TemporaryDirectory() as local_model_dir:
        # Copy the model files from S3 to local directory
        logging.info(f"Copying model files from {s3_path} to {local_model_dir}")
        copy_dir_from_s3(fs, s3_path, local_model_dir)
        logging.info("Model files copied to local directory")

        # Initialize tokenizer and model from local directory
        tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            local_model_dir, use_safetensors=True
        )
        model.gradient_checkpointing_enable()
        model.use_cache = False

        # Prepare the model for distributed training
        model = prepare_model(model)

        def preprocess_function(examples):
            prompts = [
                f"Review: {text}\nSentiment: {label}\n"
                for text, label in zip(examples["text"], examples["label"], strict=True)
            ]
            encoded = tokenizer(
                prompts, truncation=True, max_length=512, padding="max_length"
            )
            encoded["labels"] = encoded["input_ids"].copy()
            return encoded

        # Preprocess datasets
        tokenized_train = train_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names,  # pyright: ignore
        )
        tokenized_test = test_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=test_dataset.column_names,  # pyright: ignore
        )

        # Convert to PyTorch datasets
        train_dataset = tokenized_train.with_format("torch")
        test_dataset = tokenized_test.with_format("torch")

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=config["batch_size"], shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config["batch_size"], shuffle=False
        )

        # Prepare data loaders for distributed training
        train_loader = prepare_data_loader(train_loader)
        test_loader = prepare_data_loader(test_loader)

        # Training setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

        # Training loop
        model.train()
        epoch_loss = 0.0
        for epoch in range(config["num_epochs"]):
            epoch_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()

                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                # Report metrics to Ray
                ray.train.report({"training_loss": loss.item(), "epoch": epoch})

            # Evaluate on test set
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for batch in test_loader:
                    outputs = model(**batch)
                    test_loss += outputs.loss.item()
            model.train()

            # Report test loss
            ray.train.report(
                {"test_loss": test_loss / len(test_loader), "epoch": epoch}
            )

        # Save the model on the main process
        if rank == 0:
            # Save the model to local directory
            if isinstance(model, DistributedDataParallel):
                model.module.save_pretrained(local_model_dir)
            else:
                model.save_pretrained(local_model_dir)
            tokenizer.save_pretrained(local_model_dir)

            # Copy the saved model files back to S3
            checkpoint_path = f"models/checkpoints/sentiment_model/{ray.train.get_context().get_trial_id()}"
            copy_dir_to_s3(fs, local_model_dir, checkpoint_path)

    return {"final_loss": epoch_loss / len(train_loader)}


if __name__ == "__main__":
    logging.info("=== STARTING TRAINING SCRIPT ===")
    
    # Run `make ray_port_forward` ahead of this to make the ray cluster accessible
    ray_address = "ray://localhost:10001"
    logging.info(f"Connecting to Ray cluster at: {ray_address}")
    ray.init(address=ray_address, runtime_env=runtime_env)
    logging.info("Successfully connected to Ray cluster!")

    with TemporaryDirectory() as results_dir:
        # Create the trainer
        trainer = TorchTrainer(
            train_func,
            train_loop_config=train_config,
            scaling_config=scaling_config,
            run_config=RunConfig(results_dir),
        )

    # Logging something
    es = Elasticsearch([f"http://localhost:9200"])

    es.index(index="logs", document={"message": "Started training"})

    # Start training
    try:
        results = trainer.fit()
        logging.info(f"Training completed successfully! Results: {results}")
    finally:
        ray.shutdown()
        logging.info("=== TRAINING SCRIPT FINISHED ===")
