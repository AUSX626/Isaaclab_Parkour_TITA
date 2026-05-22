#!/bin/bash
# 录制全部 10 种地形的 play 视频，保存到 videos/all_terrains/
set -e

cd /home/kemove/zhr_data/project-instinct-tita/instinctlab

source ~/miniforge3/etc/profile.d/conda.sh
conda activate instinct
source /home/kemove/isaac-sim/setup_conda_env.sh
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

TASK="Instinct-Parkour-Target-Tita-Play-v0"
LOAD_RUN="20260411_143337"
VIDEO_LEN=3000
SUBDIR="all_terrains"

TERRAIN_NAMES=(
    "perlin_rough"
    "perlin_rough_stand"
    "square_gaps"
    "pyramid_stairs"
    "pyramid_stairs_high"
    "pyramid_stairs_inv"
    "pyramid_stairs_inv_high"
    "boxes"
    "mesh_boxes"
    "hf_pyramid_slope_inv"
)

for i in $(seq 0 9); do
    echo ""
    echo "========================================"
    echo " Recording terrain ${i}: ${TERRAIN_NAMES[$i]}"
    echo "========================================"
    python scripts/instinct_rl/play.py \
        --task "$TASK" \
        --load_run "$LOAD_RUN" \
        --video --video_length "$VIDEO_LEN" \
        --headless \
        --camera_env_id "$i" \
        --video_subdir "$SUBDIR"
done

echo ""
echo "All done! Videos saved to:"
echo "logs/instinct_rl/tita_parkour/${LOAD_RUN}/videos/${SUBDIR}/"
