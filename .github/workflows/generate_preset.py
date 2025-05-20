import json
import sys
from pathlib import Path

from cookiecutter.main import cookiecutter

# Get command line arguments
if len(sys.argv) < 3:
    print("Usage: python generate_charts.py <security_level> [output_base_dir]")
    sys.exit(1)

security_level = sys.argv[1]

output_base_dir = Path(sys.argv[2])

# Get the repo directory - this will be on the configurable_security branch
repo_dir = Path(".").absolute()

presets = json.loads((repo_dir / "presets.json").read_text())

# Get features for this preset
features = presets[security_level].get("features", [])

# Setup context
context = {k: True for k in features}
context["project_slug"] = "ca-k8s-infra"

# Create output directory for this preset
output_dir = (output_base_dir / security_level).absolute()
output_dir.mkdir(parents=True, exist_ok=True)

# Run cookiecutter with the current directory as template
cookiecutter(
    str(repo_dir),
    output_dir=str(output_dir),
    no_input=True,
    extra_context=context,
)
