import os
import json

DEPENDENCIES = {
    "cilium_network_policies": [],
    "egress_filtering": ["cilium_network_policies"],
}

def get_cookiecutter_context():
    # Extract variables from environment
    context_str = r"""
{{ cookiecutter | jsonify }}
"""
    return json.loads(context_str)

def main():
    context = get_cookiecutter_context()

    # Iterate through the entire context. Since we ignore all keys that are not in DEPENDENCIES, we don't have to filter out anything at this point.
    for key, value in context.items():
        if depends_on := DEPENDENCIES.get(key):
            # This key belongs to a feature with dependencies
            if not value:
                # The feature is disabled
                continue

            for dep in depends_on:
                if dep not in context:
                    raise Exception(f"'{key}' requires '{dep}' but we cannot find it in the context")
                if not context[dep]:
                    raise Exception(f"'{key}' requires '{dep}' to be enabled")

if __name__ == "__main__":
    main()
