#!/bin/bash
# ============================================================================
# One-time setup of the tree-clipping virtual environment on GACRC.
# Run this INTERACTIVELY (not as a SLURM batch job):
#     bash setup_clip_env.sh
# ============================================================================
set -euo pipefail

# Where the venv will live. HOME is used (not scratch) so it persists --
# /scratch is auto-purged after ~30 days of inactivity. This matches the
# VENV path in jkw_pytree_clip.sh: /home/$(whoami)/venv
VENV="/home/$(whoami)/venv"

# Load the SAME Python module the batch jobs use.
module load Python/3.11.5-GCCcore-13.2.0

# Keep pip's download cache off your /home quota (open3d is large). The cache
# is transient and regenerable, so scratch is fine for it.
export PIP_CACHE_DIR="/scratch/$(whoami)/.pip_cache"

# Create the venv only if it doesn't already exist.
if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment at: $VENV"
    python -m venv "$VENV"
else
    echo "Virtual environment already exists at: $VENV"
fi

# Activate and install.
source "$VENV/bin/activate"

# Now installing all packages need for the treeclip -> treeQSM -> tree skeleton pipeline! 
python -m pip install --upgrade pip
pip install numpy open3d scikit-learn

echo ""
echo "Done. Virtual environment ready at: $VENV"
echo "In your batch script (jkw_pytree_clip.sh), activate it with:"
echo "    module load Python/3.11.5-GCCcore-13.2.0"
echo "    source $VENV/bin/activate"