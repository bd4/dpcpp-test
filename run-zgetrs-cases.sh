#!/bin/bash

exe=$(dirname $0)/build-intelone/batched_zgetrs
echo exe=$exe

($exe  140 1 384
 $exe  256 1 512
 $exe  512 1 512
 $exe  768 1 256
 $exe 1024 1 128) \
 | tee "zgetrs-log-$(hostname).txt"
