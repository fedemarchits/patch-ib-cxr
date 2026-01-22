#!/bin/bash
# train.sh

# Default config if $1 is empty
CONFIG=${1:-model_a_baseline.yaml}
# Shift the arguments so that $@ now contains everything EXCEPT the config name
shift 

echo "🚀 Starting Training..."
echo "📄 Config: configs/$CONFIG"
echo "additional flags: $@"

# $@ passes ALL remaining arguments to the python script
python3 train.py --config "configs/$CONFIG" "$@"