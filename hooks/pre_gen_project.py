import json

DEPENDENCIES: dict[str, list[str]] = {
    "network_policies": [],
    "external_secrets": [],
    "rbac": ["external_secrets"],
    "honeypot": [],
    "teams": ["rbac"],
    "egress_filtering": ["network_policies"],
    "tetragon": [],
    "file_integrity": ["tetragon"],
    "security_monitoring": ["tetragon"],
    "tripwires": ["file_integrity"],
}

# List of mutually incompatible measures
INCOMPATIBLE: list[tuple] = []


def get_cookiecutter_context():
    # Extract variables from environment
    context_str = r"""
{{ cookiecutter | jsonify }}
"""
    return json.loads(context_str)


def main():
    context = get_cookiecutter_context()

    # Iterate through the entire context.
    # Since we ignore all keys that are not in DEPENDENCIES, we don't have to filter out anything at this point.
    for key, value in context.items():
        if depends_on := DEPENDENCIES.get(key):
            # This key belongs to a feature with dependencies
            if not value:
                # The feature is disabled
                continue

            for dep in depends_on:
                if dep not in context:
                    raise ValueError(
                        f"'{key}' requires '{dep}' but we cannot find it in the context"
                    )
                if not context[dep]:
                    raise ValueError(f"'{key}' requires '{dep}' to be enabled")

    for incompatible_measures in INCOMPATIBLE:
        if sum(context.get(measure, False) for measure in incompatible_measures) > 1:
            raise ValueError(
                f"These security measures are incompatible: {', '.join(incompatible_measures)}"
            )


if __name__ == "__main__":
    main()
