import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from azure.ai.ml import Input, MLClient, Output, command, dsl
from azure.ai.ml.entities import Command, PipelineJobSettings
from azure.identity import DefaultAzureCredential
from rich.console import Console
from rich.panel import Panel

from src.orchestrate.config import OrchestratorConfig
from src.orchestrate.loader import load_orchestrator_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("orchestrator")
console = Console()

ROOT = Path(__file__).parent
CONFIG: OrchestratorConfig = load_orchestrator_config(
    "settings/orchestrator_config.yaml"
)


# ----- git SHA configuration -----
def resolve_git_sha(allow_dirty: bool = False) -> str:
    # CI first: GitHub Actions, Azure DevOps, or your own override
    sha = os.getenv("GITHUB_SHA")
    if sha:
        return sha

    # local fallback, if environment not set
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"]) != 0
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise SystemExit("Could not determine git SHA; set GIT_SHA.")

    if dirty:
        if not allow_dirty:
            raise SystemExit("Working tree is dirty; commit or pass --allow-dirty.")
        return f"{sha}-dirty"
    return sha


# ----- AML client and environment -------
def get_client() -> MLClient:
    return MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=CONFIG.workspace.subscription_id,
        resource_group_name=CONFIG.workspace.resource_group,
        workspace_name=CONFIG.workspace.workspace_name,
    )


# ------ Base Job Config ----------
def _base_job_kwargs(name: str, description: str) -> dict:

    return {
        "display_name": name,
        "description": description,
        "environment": CONFIG.environment.full_name,
        "compute": CONFIG.compute.compute_cluster,
        "experiment_name": CONFIG.pipeline.experiment_name,
        "code": "./src",
        "environment_variables": {"PYTHONPATH": "./"},
    }


# ------ Job Builders ---------
def data_versioning_component() -> Command:

    return command(
        **_base_job_kwargs(
            "data-versioning-gate", "Fetch, clean, split, and version data"
        ),
        command=(
            "python -m gates.data_versioning_gate "
            "--raw-data ${{inputs.raw_data}} "
            "--output-training-path ${{outputs.processed_data}}"
        ),
        inputs={
            "raw_data": Input(type="uri_file", path=CONFIG.data.name),
        },
        outputs={
            "processed_data": Output(
                type="uri_folder",
                mode="rw_mount",
                name=CONFIG.data.output_asset_name,
            )
        },
    )


def drift_detection_component() -> Command:
    return command(
        **_base_job_kwargs(
            "drift-detection-gate", "Statistical drift check against baseline"
        ),
        command=(
            "python -m gates.drift_detection_gate "
            "--new-data-version ${{inputs.processed_data}} "
            "--gold-data-version ${{inputs.gold_data}} "
            "--drift-output-path ${{outputs.drift_output}}"
        ),
        inputs={
            "gold_data": Input(type="uri_file", path=CONFIG.data.name),
            "processed_data": Input(
                type="uri_folder",
            ),
        },
        outputs={"drift_output": Output(type="uri_folder", mode="rw_mount")},
    )


def model_training_component() -> Command:
    return command(
        **_base_job_kwargs("model-training-gate", "Model fit gate to train new model"),
        command=(
            "python -m gates.model_training_gate "
            "--drift-detected ${{inputs.drift_detected}} "
            "--training-data ${{inputs.processed_data}} "
            "--git-sha ${{inputs.git_sha}} "
            "--final-run ${{outputs.final_run_id}}"
        ),
        inputs={
            "processed_data": Input(type="uri_folder"),
            "drift_detected": Input(type="uri_folder"),
            "git_sha": Input(type="string"),
        },
        outputs={"final_run_id": Output(type="uri_folder", mode="rw_mount")},
    )


def model_promotion_component() -> Command:
    return command(
        **_base_job_kwargs(
            "model-promotion-gate",
            "Model promotion gate to assess challenger promotion",
        ),
        command=(
            "python -m gates.model_promotion_gate "
            "--final-run-id ${{inputs.final_run_id}} "
            "--processed-data ${{inputs.processed_data}} "
            "--decision-output ${{outputs.promotion_decision}}"
        ),
        inputs={
            "final_run_id": Input(type="uri_folder"),
            "processed_data": Input(type="uri_folder"),
        },
        outputs={"promotion_decision": Output(type="uri_folder", mode="rw_mount")},
    )


def model_deployment_component() -> Command:
    return command(
        **_base_job_kwargs(
            "model-deployment-get",
            "Deploys model to Azure ML endpoint",
        ),
        command=(
            "python -m gates.model_deployment_gate "
            "--decision-input ${{inputs.promotion_data}}"
        ),
        inputs={"promotion_data": Input(type="uri_folder")},
    )


@dsl.pipeline(
    description="Gated classification pipeline",
    experiment_name=CONFIG.pipeline.experiment_name,
    default_compute=CONFIG.compute.compute_cluster,
)
def build_pipeline(raw_data: Input, gold_data: Input, git_sha: str):
    data_step = data_versioning_component()(
        raw_data=raw_data,
    )

    drift_step = drift_detection_component()(
        gold_data=gold_data,
        processed_data=data_step.outputs.processed_data,
    )

    model_step = model_training_component()(
        processed_data=data_step.outputs.processed_data,
        drift_detected=drift_step.outputs.drift_output,
        git_sha=git_sha,
    )

    promotion_step = model_promotion_component()(
        final_run_id=model_step.outputs.final_run_id,
        processed_data=data_step.outputs.processed_data,
    )

    _deployment_step = model_deployment_component()(
        promotion_data=promotion_step.outputs.promotion_decision
    )

    return {"drift_output": drift_step.outputs.drift_output}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-dirty", action="store_true")
    cli = parser.parse_args()

    ml_client = get_client()
    git_sha = resolve_git_sha(allow_dirty=cli.allow_dirrty)
    # The same data is the gold and new data to test drift
    pipeline_job = build_pipeline(
        raw_data=Input(type="uri_file", path=CONFIG.data.name),
        gold_data=Input(type="uri_file", path=CONFIG.data.name),
        git_sha=git_sha,
    )
    pipeline_job.settings = PipelineJobSettings(
        force_rerun=True, default_compute=CONFIG.compute.compute_cluster
    )

    try:
        returned_job = ml_client.jobs.create_or_update(pipeline_job)
        job_url = returned_job.studio_url
        console.print("\n[bold cyan]▶ Pipeline submitted[/bold cyan]")
        console.print(f"  Job ID : {returned_job.name}")
        console.print(f"  Studio : [link={job_url}]{job_url}[/link]")

        # Stream logs — blocks until pipeline completes
        ml_client.jobs.stream(returned_job.name)

        console.print(Panel("[bold green]Pipeline complete[/bold green]"))

    except RuntimeError as exc:
        console.print(
            Panel(f"[bold red]Pipeline halted:\n{exc}[/bold red]", title="FAILED")
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
