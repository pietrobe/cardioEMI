#!/bin/bash

# Define the container file
CONTAINER="dolfinx_v0.7.1.sif"

# Check if the container exists
if [ ! -f "$CONTAINER" ]; then
    echo "Error: Container $CONTAINER not found!"
    exit 1
fi

# Check if the correct number of arguments is provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <num_processes> <input_file>"
    exit 1
fi

# Read command-line arguments
N="$1"         # Number of processes
INPUT_FILE="$2" # Input file

# Define the required bind mounts
BIND_VIM="--bind /usr/bin/vim:/usr/bin/vim"
BIND_LIBGPM="--bind /home/kit/ibt/au0395/.local/share/enroot/pyxis_kaskade7/usr/lib/x86_64-linux-gnu/libgpm.so.2:/usr/lib/x86_64-linux-gnu/libgpm.so.2"

# Run the container, install multiphenicsx silently, and execute the simulation
apptainer exec $BIND_VIM $BIND_LIBGPM "$CONTAINER" bash -c "
    pip install -q multiphenicsx@git+https://github.com/multiphenics/multiphenicsx.git@dolfinx-v0.7.1 > /dev/null 2>&1
    mpirun -n $N python3 -u main.py $INPUT_FILE
"

