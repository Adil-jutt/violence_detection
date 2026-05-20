#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python -m grpc_tools.protoc \
    -I. \
    --python_out=. \
    --grpc_python_out=. \
    violence.proto
echo "Done — violence_pb2.py and violence_pb2_grpc.py generated."
