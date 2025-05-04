import json
import logging
import os
from datetime import datetime

from fastapi import FastAPI, Header, HTTPException, Request, Response
from minio import Minio

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/var/log/honeypot/honeypot.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("honeypot-monitor")

# Environment variables
ENVIRONMENT = os.environ.get("ENVIRONMENT", "dev")
ALERT_TOKEN = os.environ.get("ALERT_TOKEN", "honeypot-alert-token")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")

# Initialize the FastAPI app
app = FastAPI(title="Honeypot Monitor")

# Initialize MinIO client
minio_client = Minio(
    f"minio.minio-internal-{ENVIRONMENT}.svc.cluster.local:9000",
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)

# Known honeypot files and their sensitivity
HONEYPOT_FILES = {
    "models/.credentials/aws-credentials.json": "high",
    "datasets/team-confidential/customer-data.json": "medium",
}


@app.post("/alert")
async def handle_alert(request: Request):
    """Handle MinIO event notifications for honeypot access."""
    try:
        payload = await request.json()

        # Extract event details
        records = payload.get("Records", [])
        if not records:
            return {"status": "error", "message": "No records in payload"}

        for record in records:
            s3 = record.get("s3", {})
            bucket = s3.get("bucket", {}).get("name", "unknown")
            object_key = s3.get("object", {}).get("key", "unknown")
            event_name = record.get("eventName", "unknown")
            source_ip = record.get("requestParameters", {}).get(
                "sourceIPAddress", "unknown"
            )
            user_agent = record.get("requestParameters", {}).get("userAgent", "unknown")

            # Full path for checking against honeypot files
            full_path = f"{bucket}/{object_key}"

            # Check if this is a known honeypot file
            sensitivity = HONEYPOT_FILES.get(full_path, "low")

            # Log the access
            logger.warning(
                f"HONEYPOT ALERT: {event_name} on {full_path} "
                f"(Sensitivity: {sensitivity}) from IP: {source_ip} UA: {user_agent}"
            )

        return {"status": "success", "message": "Alert processed"}

    except Exception as e:
        logger.error(f"Error processing alert: {str(e)}")
        return {"status": "error", "message": str(e)}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}
