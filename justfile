set dotenv-load := true

# Submit the DSL pipeline to Azure ML
run:
    uv run orchestrator.py --allow-dirty

# Create you Azure ML environment
environment:
    uv run infra/environment/create_environment.py

# Sync and install dependencies
sync:
    uv sync --frozen
