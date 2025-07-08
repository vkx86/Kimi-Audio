#!/bin/bash

# cp ./.devcontainer/sources.txt /etc/apt/sources.list

apt-get update && apt-get install -y --no-install-recommends \
    git nano wget