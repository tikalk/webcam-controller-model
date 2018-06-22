#!/usr/bin/env bash

CURL=`which curl`
DATA_DIR='./data'
[[ -z "${CURL}" ]] && echo "Please intstall curl and rerun $0"

mkdir ${DATA_DIR}
pushd ${DATA_DIR}
[[ -f ${DATA_DIR}/ys.csv ]] || curl -O http://fsdi-jun18-ml-static.s3-website-eu-west-1.amazonaws.com/webcam-controller-model/data/ys.csv
[[ -f ${DATA_DIR}/xs.csv ]] || curl -O http://fsdi-jun18-ml-static.s3-website-eu-west-1.amazonaws.com/webcam-controller-model/data/xs.csv
popd
