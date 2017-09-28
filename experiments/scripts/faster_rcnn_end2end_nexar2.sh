#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
PHASE=$3
CAFFEMODEL=$4

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

TRAIN_IMDB="nexar2_train_40k"
TRAIN_ITERS=30000

TEST_IMDB="nexar2_val_10k"

case $PHASE in
  train)
    LOG="experiments/logs/faster_rcnn_end2end_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
    exec &> >(tee -a "$LOG")
    echo Logging output to "$LOG"

    time ./tools/train_net.py --gpu ${GPU_ID} \
      --solver models/nexar2/${NET}/faster_rcnn_end2end/solver.prototxt \
      --weights ${CAFFEMODEL} \
      --imdb ${TRAIN_IMDB} \
      --iters ${TRAIN_ITERS} \
      --cfg experiments/cfgs/faster_rcnn_end2end.yml \
      ${EXTRA_ARGS}
    ;;
  test)
    time ./tools/test_net.py --gpu ${GPU_ID} \
      --def models/nexar2/${NET}/faster_rcnn_end2end/test.prototxt \
      --net ${CAFFEMODEL} \
      --imdb ${TEST_IMDB} \
      --cfg experiments/cfgs/faster_rcnn_end2end.yml \
      ${EXTRA_ARGS}
      ;;
  *)
    echo "Wrong phase (must be test or train)"
    exit
    ;;
esac