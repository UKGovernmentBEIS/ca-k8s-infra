FROM localhost:5001/training:latest

WORKDIR /app

# Install dependencies
RUN pip install pytest

# Set PYTHONPATH to include the app directory
ENV PYTHONPATH=/app:$PYTHONPATH

# Default command to run tests
CMD ["python", "-m", "pytest", "training/tests/test_train.py", "-v"]
