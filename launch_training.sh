#!/usr/bin/env bash

TRAIN_DATASET='/home/felix/Projects/tool-detection/dataset_parsing/det_ready/bypass_margin20px/bypass_cleaned_reduced_margin20px/ds_bypass_tools_segs_to_bb_margin20px_retinanet_train.csv'
TEST_DATASET='/home/felix/Projects/tool-detection/dataset_parsing/det_ready/bypass_margin20px/bypass_cleaned_reduced_margin20px/ds_bypass_tools_segs_to_bb_margin20px_retinanet_test.csv'

NET ="vgg16-ssd"
BASE_NET="models/vgg16_reducefc.pth"
BATCH_SIZE=24
NUM_EPOCHS=200
SCHEDULER="multi-step"
MILESTONES="120,160"



echo "Training ${NET} for ${NUM_EPOCHS} with batch size of ${BATCH_SIZE}"
python train_ssd.py \
    --train_dataset=${TRAIN_DATASET} \
    --validation_dataset=${TEST_DATASET} \
    --net=${NET} \
    --base_net=${BASE_NET} \
    --batch_size=${BATCH_SIZE} \
    --num_epochs=${NUM_EPOCHS} \
    --scheduler=${SCHEDULER} \
    --milestones=${MILESTONES}