#!/bin/bash

MODEL="ResNet18"
DATASET="cifar10"
TRAINER="SL"
NOTE="RCE"
TRIGGER="badnets"
RATE="010"

echo "Start in $SECONDS seconds"

python train.py --config conf/${MODEL}_${DATASET}_${TRAINER}_${TRIGGER}_rate${RATE}.yml | tee ${MODEL}_${TRAINER}_${DATASET}_${TRIGGER}_Rate${RATE}_${NOTE}.log

echo "Completed in $SECONDS seconds"
