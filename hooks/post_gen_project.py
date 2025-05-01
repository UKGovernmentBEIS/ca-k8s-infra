#!/usr/bin/env python
import os
import shutil
import yaml
import json
import logging

log = logging.getLogger(__name__)


def copy_overlay(project_dir: str, overlay_name: str):
    """
    Merge everything under overlays/<overlay_name> into project_dir,
    preserving the relative sub-folders.
    """
    src_root = os.path.join(project_dir, "overlays", overlay_name)
    if not os.path.isdir(src_root):
        return

    for root, dirs, files in os.walk(src_root):
        # compute where in project_dir this should land
        rel = os.path.relpath(root, src_root)  # e.g. "" or "training-infra/templates"
        dest_root = os.path.join(project_dir, rel)

        os.makedirs(dest_root, exist_ok=True)
        for fname in files:
            src_file = os.path.join(root, fname)
            dst_file = os.path.join(dest_root, fname)
            # overwrite any base file with the overlay
            shutil.copy2(src_file, dst_file)


def rm_if_exists(path: str):
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.isfile(path):
        os.remove(path)


def rm_empty_files(path: str):
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for fname in files:
                if fname.endswith(".yaml"):
                    full_path = os.path.join(root, fname)
                    with open(full_path, 'rt') as f:
                        try:
                            content = yaml.safe_load_all(f.read())
                            if all(doc is None for doc in content):
                                log.info("Removing empty file:", full_path)
                                os.remove(full_path)
                        except yaml.YAMLError:
                            # We cannot parse Helm template syntax
                            continue


def get_cookiecutter_context():
    # Extract variables from environment
    context_str = r"""
{{ cookiecutter | jsonify }}
"""
    return json.loads(context_str)


def main():
    # Cookiecutter runs this script with cwd = the generated project root
    project_dir = os.getcwd()

    # Read the flags that Cookiecutter set for us
    # Merge in any overlays the user asked for
    context = get_cookiecutter_context()
    includes = [key for key in context if key != "project_slug" and context[key]]
    for include in includes:
        log.info(f"Processing overlay: {include}")
        copy_overlay(project_dir, include)

    # Remove any empty YAML files
    rm_empty_files(project_dir)

    # Now get rid of the entire overlays/ folder
    rm_if_exists(os.path.join(project_dir, "overlays"))


if __name__ == "__main__":
    main()
