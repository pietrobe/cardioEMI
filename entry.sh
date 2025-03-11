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
    echo "Usage: $0 <num_processes> <input_file> [--param value] ..."
    exit 1
fi

# Read command-line arguments
N="$1"         # Number of processes
INPUT_FILE="$2" # Input file
shift 2  # Shift past the first two arguments

# Create a temporary YAML file based on input.yml
TMP_FILE=$(mktemp --suffix=.yml)
cp "$INPUT_FILE" "$TMP_FILE"

# Extract default out_name from input file
OUT_NAME=$(grep '^out_name' "$INPUT_FILE" | awk -F ': ' '{print $2}' | tr -d '"')

# Process the dynamic parameters
MODIFIED=false
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --*)
            PARAM_NAME="${1#--}"  # Remove leading --
            PARAM_VALUE="$2"
            sed -i "s|^$PARAM_NAME\s*: .*|$PARAM_NAME: $PARAM_VALUE|" "$TMP_FILE"
            shift 2
            MODIFIED=true
            
            # Update OUT_NAME if it's being changed
            if [ "$PARAM_NAME" == "out_name" ]; then
                OUT_NAME="$PARAM_VALUE"
            fi
            ;;
        *)
            echo "Invalid argument: $1"
            exit 1
            ;;
    esac
done

# Ensure output directory exists
mkdir -p output

# Save the modified input file only if parameters were changed
if [ "$MODIFIED" = true ]; then
    FINAL_INPUT_FILE="output/input-${OUT_NAME}.yml"
    cp "$TMP_FILE" "$FINAL_INPUT_FILE"
else
    FINAL_INPUT_FILE="$TMP_FILE"
fi

# Define the required bind mounts
BIND_VIM="--bind /usr/bin/vim:/usr/bin/vim"
BIND_LIBGPM="--bind /home/kit/ibt/au0395/.local/share/enroot/pyxis_kaskade7/usr/lib/x86_64-linux-gnu/libgpm.so.2:/usr/lib/x86_64-linux-gnu/libgpm.so.2"

# Run the container, install multiphenicsx silently, and execute the simulation
apptainer exec $BIND_VIM $BIND_LIBGPM "$CONTAINER" bash -c "
    pip install -q multiphenicsx@git+https://github.com/multiphenics/multiphenicsx.git@dolfinx-v0.7.1 > /dev/null 2>&1
    mpirun -n $N python3 -u main.py $FINAL_INPUT_FILE
"

# Clean up temporary file
rm "$TMP_FILE"
