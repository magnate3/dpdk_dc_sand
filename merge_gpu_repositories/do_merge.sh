#!/bin/bash
set -e  # Stop on errors instead of bumbling along.

COMMON_NAME=katgpucbf
FGPU_NAME=fgpu
XGPU_NAME=xbgpu

# Fresh clone of both repos.
git clone git@github.com:ska-sa/katfgpu.git
git clone git@github.com:ska-sa/katxgpu.git

cd katfgpu
# Get rid of the nvvp files from katfgpu
git filter-repo --path-glob '*.nvvp' --invert-paths
git rm -r 3rdparty
git commit -am "Remove submodule since katxgpu will also pull it in and cause conflict"
cd ..

# This trick makes the prebeamformer_reorder branch available in the
# new repo.
cd katxgpu
git checkout prebeamform_reorder
git checkout main
cd ..

# Create a new, combined repo
git init $COMMON_NAME
cd $COMMON_NAME
git branch -m master main

# Start pulling in katfgpu's history
git remote add -f $FGPU_NAME ../katfgpu/
git merge $FGPU_NAME/master --allow-unrelated-histories --no-edit

# Move the C++ files into the same folder as the Python source.
# Then move all the source code to a temp location to make the merge of katxgpu easier. 
# Move the docs to their proper subfolder (katxgpu's is called docs so there's no collission).
# Move the scratch stuff to a temporary place to prevent collissions with katxgpu.
# All the individual files in the root folder get moved to a misc folder for later manual procesing.
git filter-repo --path-rename src/:katfgpu/_katfgpu/ \
                --path-rename katfgpu/:src_$COMMON_NAME/$FGPU_NAME/ \
                --path-rename test/:test_$FGPU_NAME/ \
                --path-rename doc/:doc/$FGPU_NAME/ \
                --path-rename scratch/:scratch_$FGPU_NAME/ \
                --path-rename .dockerignore:misc/$FGPU_NAME/.dockerignore \
                --path-rename .flake8:misc/$FGPU_NAME/.flake8 \
                --path-rename .gitignore:misc/$FGPU_NAME/.gitignore \
                --path-rename .pre-commit-config.yaml:misc/$FGPU_NAME/.pre-commit-config.yaml \
                --path-rename .pydocstyle.ini:misc/$FGPU_NAME/.pydocstyle.ini \
                --path-rename Dockerfile:misc/$FGPU_NAME/Dockerfile \
                --path-rename README.md:misc/$FGPU_NAME/README.md \
                --path-rename mypy.ini:misc/$FGPU_NAME/mypy.ini \
                --path-rename pyproject.toml:misc/$FGPU_NAME/pyproject.toml \
                --path-rename pytest.ini:misc/$FGPU_NAME/pytest.ini \
                --path-rename requirements-dev.txt:misc/$FGPU_NAME/requirements-dev.txt \
                --path-rename requirements.txt:misc/$FGPU_NAME/requirements.txt \
                --path-rename setup.py:misc/$FGPU_NAME/setup.py \
                --force
# We need to use --force argument because it thinks we are not on a fresh clone. Which we aren't.

# Pull in katxgpu's history.
git branch prebeamform_reorder
git remote add -f $XGPU_NAME ../katxgpu/
git merge $XGPU_NAME/main --allow-unrelated-histories --no-edit
git checkout prebeamform_reorder
git merge $XGPU_NAME/prebeamform_reorder --allow-unrelated-histories --no-edit
git checkout main

# Same thing happens to the C++ source - join the Python files first.
# Then move everything to its appropriate place in the src/ tree.
# Then we can shift katfgpu's stuff into its rightful place alongside.
# The same thing with docs and scratch.
# Misc files also get to their own folder, those will need to be manually combined.
git filter-repo --path-rename src/:katxgpu/_katxgpu/ \
                --path-rename katxgpu/:src/$COMMON_NAME/$XGPU_NAME/ \
                --path-rename src_$COMMON_NAME/:src/$COMMON_NAME/ \
                --path-rename test/:test/$XGPU_NAME/ \
                --path-rename test_$FGPU_NAME/:test/$FGPU_NAME/ \
                --path-rename docs/:doc/$XGPU_NAME/ \
                --path-rename scratch/:scratch/$XGPU_NAME/ \
                --path-rename scratch_$FGPU_NAME/:scratch/$FGPU_NAME/ \
                --path-rename Dockerfile:misc/$XGPU_NAME/Dockerfile \
                --path-rename Jenkinsfile:misc/$XGPU_NAME/Jenkinsfile \
                --path-rename MANIFEST.in:misc/$XGPU_NAME/MANIFEST.in \
                --path-rename README.md:misc/$XGPU_NAME/README.md \
                --path-rename pyproject.toml:misc/$XGPU_NAME/pyproject.toml \
                --path-rename pytest.ini:misc/$XGPU_NAME/pytest.ini \
                --path-rename requirements-dev.txt:misc/$XGPU_NAME/requirements-dev.txt \
                --path-rename requirements.txt:misc/$XGPU_NAME/requirements.txt \
                --path-rename setup.py:misc/$XGPU_NAME/setup.py \
                --force

