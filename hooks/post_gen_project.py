#!/usr/bin/env python
import os
import shutil

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

def main():
    # Cookiecutter runs this script with cwd = the generated project root
    project_dir = os.getcwd()

    # Read the flags that Cookiecutter set for us
    # Merge in any overlays the user asked for
    includes = [key for key in '{{ ' '.join(cookiecutter.keys()) }}'.split(' ') if key.startswith("include_")]
    for include in includes:
        copy_overlay(project_dir, include[8:])

    # Now get rid of the entire overlays/ folder
    rm_if_exists(os.path.join(project_dir, "overlays"))

if __name__ == "__main__":
    main()
