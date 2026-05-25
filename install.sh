#!/usr/bin/env bash

set -e

source "$(dirname -- "${BASH_SOURCE[0]}")/lib.include.sh"

prepare_runtime_environment upgrade
