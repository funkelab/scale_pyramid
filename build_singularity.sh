#!/bin/sh
set -ex
docker build -t scale_pyramid .
singularity build scale_pyramid.sif docker-daemon://scale_pyramid:latest
