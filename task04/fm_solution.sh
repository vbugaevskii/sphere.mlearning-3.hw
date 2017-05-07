#!/bin/bash

LIBFM_PATH='libfm-1.42.src/bin/libFM'

TRAIN_PATH='data/train.timestamp.txt.libfm'
TEST_PATH='data/test.timestamp.txt.libfm'

ITER=100
LATENT=10

$LIBFM_PATH -task r -train $TRAIN_PATH -test $TEST_PATH \
            -dim "1,1,$LATENT" -iter $ITER -method mcmc \
            -out "output_$LATENT-$ITER.libfm"

python convert_output.py "output_$LATENT-$ITER.libfm" "submission_02_$LATENT-$ITER.txt"
