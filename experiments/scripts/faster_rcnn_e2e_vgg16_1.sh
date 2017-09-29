#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1

EXP_ID=1
TRAIN_IMDB="nexar2_train_40k"
TRAIN_ITERS=80000
SOLVER=models/nexar2/VGG16/faster_rcnn_end2end/solver_1.prototxt
WEIGHTS=output/faster_rcnn_end2end/nexar2_train_40k/vgg16_faster_rcnn_iter_10000.caffemodel # trained with lr=1e-3
CONFIG=experiments/cfgs/faster_rcnn_end2end_nexar2.yml

LOG="experiments/logs/faster_rcnn_end2end_${EXP_ID}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu ${GPU_ID} \
  --solver ${SOLVER} \
  --weights ${WEIGHTS} \
  --imdb ${TRAIN_IMDB} \
  --iters ${TRAIN_ITERS} \
  --cfg ${CONFIG} \
  --set EXP_DIR faster_rcnn_end2end_${EXP_ID} TRAIN.SNAPSHOT_ITERS 10000