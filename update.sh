#!/usr/bin/env bash

set -e

# Load the current function library before mutating the checkout. This preserves
# fail-loud handling for removed bootstrap environment variables.
source "$(dirname -- "${BASH_SOURCE[0]}")/lib.include.sh"

# Pull the latest changes via Git.
echo "[OneTrainer] Updating OneTrainer to latest version from Git repository..."
git pull

# Load the newest version of the function library.
source "lib.include.sh"

# Prepare runtime and upgrade all dependencies to latest compatible version.
prepare_runtime_environment upgrade
