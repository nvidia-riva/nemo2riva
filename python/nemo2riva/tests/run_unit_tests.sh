#!/usr/bin/env bash
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# to run locally with no environmental modifactions please mount the same volume as used by CI.
# currently this is one on linux via the following:
# sudo mount -t nfs 10.31.241.13:/mnt/tank/datasets/jarvis_speech_ci/ /mnt/nvdl/datasets/jarvis_speech_ci


set -euo pipefail
set -o xtrace

RUNNER_GPUS=${RUNNER_GPUS:-1}
THIS_DIR=$(cd $(dirname $0); pwd)
RIVA_DIR=$(cd ${THIS_DIR}/../../..; pwd)
NEMO_IMAGE_NAME=${NEMO_IMAGE_NAME:-nvcr.io/nvidia/nemo:22.04}
UNIT_TEST_CONTAINER_NAME=${UNIT_TEST_CONTAINER_NAME:-nemo2riva-unit-test}
DATADIR=${DATADIR:-/mnt/nvdl/datasets/}

docker rm -f ${UNIT_TEST_CONTAINER_NAME} || true
EXTRA_ARGS="--config_only"
NV_DOCKER_ARGS="curl -s \"http://localhost:3476/docker/cli?dev=${RUNNER_GPUS//,/+}\" || echo --gpus 1"
echo "Riva dir: ${RIVA_DIR}"

docker run $(eval ${NV_DOCKER_ARGS}) --init --label RUNNER_ID=${RUNNER_ID:-0} --name ${UNIT_TEST_CONTAINER_NAME} -d -i -v $DATADIR:/mnt/nvdl/datasets ${NEMO_IMAGE_NAME} /bin/bash

docker exec ${UNIT_TEST_CONTAINER_NAME} bash -c "mkdir -p /riva/python"
docker cp ${RIVA_DIR}/VERSION ${UNIT_TEST_CONTAINER_NAME}:/riva/VERSION
docker cp ${RIVA_DIR}/python/nemo2riva/ ${UNIT_TEST_CONTAINER_NAME}:/riva/python/nemo2riva

docker exec ${UNIT_TEST_CONTAINER_NAME} bash -c "find /riva"
docker exec ${UNIT_TEST_CONTAINER_NAME} bash -c "cd /riva/python/nemo2riva && pip3 install -e ."
docker exec ${UNIT_TEST_CONTAINER_NAME} bash -c "cd /riva/python/nemo2riva/tests && pytest -s --junitxml=report.xml"
docker rm -f ${UNIT_TEST_CONTAINER_NAME} || true
