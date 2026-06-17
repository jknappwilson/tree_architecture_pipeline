#!/bin/bash
#SBATCH --job-name=treeclip
#SBATCH --partition=batch
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --output=treeclip_%j.out
#SBATCH --error=treeclip_%j.err
#SBATCH --mail-user=jrk88473@uga.edu
#SBATCH --mail-type=ALL

set -euo pipefail

############################
# 1) Environment and paths
############################
# ROOT is built from the running user's scratch dir so this works for any user
# on GACRC. Replace "$(whoami)" with a hardcoded ID if you ever need to.
ROOT="/scratch/$(whoami)/treeclip_proj"
VENV="/home/$(whoami)/venv"

module load Python/3.11.5-GCCcore-13.2.0
source $VENV/bin/activate


# Input lives here; output folder gets relocated up into OUT_PARENT.
PLY_DIR="$ROOT/ply_folder/ply_rows"
OUT_PARENT="$ROOT/ply_folder"

############################
# 2) Choose which point cloud to clip
############################
# Just change this filename to clip a different cloud (file must be in PLY_DIR).
# point clouds must be in .ply format
PLY_NAME="germ_2024_b3_col19_col20.ply"

ply="$PLY_DIR/$PLY_NAME"
base="$(basename "$ply" .ply)"

echo "Running tree clipping on HPC node...stand by"
echo "Start time: $(date)"
echo "Input: $ply"

if [ ! -f "$ply" ]; then
    echo "ERROR: input file not found: $ply"
    exit 1
fi

############################
# 3) Run the clipping pipeline
############################
# Set --rows / --expected-per-row to match THIS cloud's layout.
python jkw_pytree_clip_pipeline.py \
  "$ply" \
  --rows 2 --expected-per-row 2,2 \
  --canopy-min-h 1.20 \
  --save-ground-dist 0.12 \
  --voxel-canopy 0.06 \
  --clean-ground \
  --clean-shadow-floor 0.20 \
  --preset orchard_loose \
  --classify-chunk-m 5 \
  --cc-max-points 2000000 \
  --cc-ds 0.02 \
  --cc-map-back-chunk 2000000

# The pipeline writes output next to the INPUT file, i.e. inside PLY_DIR.
src_dir="$PLY_DIR/${base}_canopy_clusters"
dst_dir="$OUT_PARENT/${base}_canopy_clusters"

if [ ! -d "$src_dir" ]; then
    echo "ERROR: expected output folder not found: $src_dir"
    exit 1
fi

############################
# 4) Convert each per-tree .ply -> .txt (X Y Z R G B), then drop the .ply
############################
python - "$src_dir" <<'PYEOF'
import sys, os, glob
import numpy as np
import open3d as o3d

out_dir = sys.argv[1]
ply_files = sorted(glob.glob(os.path.join(out_dir, "*.ply")))
for f in ply_files:
    try:
        pcd = o3d.io.read_point_cloud(f)
        pts = np.asarray(pcd.points)
        if pcd.has_colors():
            cols = np.asarray(pcd.colors)                 # open3d stores RGB as floats in [0, 1]
            cols = np.clip(np.round(cols * 255.0), 0, 255) # scale to 0-255 ints (CloudCompare-friendly)
            arr = np.hstack([pts, cols])                   # X Y Z R G B
            fmt = ["%.6f", "%.6f", "%.6f", "%d", "%d", "%d"]
        else:
            arr = pts                                      # source has no color -> XYZ only
            fmt = ["%.6f", "%.6f", "%.6f"]
        txt_path = os.path.splitext(f)[0] + ".txt"
        np.savetxt(txt_path, arr, fmt=fmt)
        os.remove(f)   # <-- comment this line out to KEEP the .ply alongside the .txt
        print(f"  converted {os.path.basename(f)} -> {os.path.basename(txt_path)} "
              f"({pts.shape[0]} pts, {arr.shape[1]} cols)")
    except Exception as e:
        print(f"  WARNING: failed to convert {os.path.basename(f)}: {e}")
PYEOF

############################
# 5) Relocate the output folder up into ply_folder/
############################
mkdir -p "$OUT_PARENT"
if [ -d "$dst_dir" ]; then
    echo "Note: $dst_dir already exists -- replacing it."
    rm -rf "$dst_dir"
fi
mv "$src_dir" "$dst_dir"

echo "[$(date)] Finished tree clipping! Output -> $dst_dir -- double check trees are correct!! Good job :)"
