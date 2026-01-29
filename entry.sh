#!/bin/bash

LOGFILE="sim.log"

# 1. Start Colima if not running
if ! colima status | grep -q "Running"; then
  echo "Starting Colima..." | tee "$LOGFILE"
  colima start >> "$LOGFILE" 2>&1
fi

# 2. Parse --mesh argument
if [[ "$1" != "--mesh" || -z "$2" ]]; then
  echo "Usage: $0 --mesh <mesh_basename>" | tee -a "$LOGFILE"
  exit 1
fi

MESH_NAME="$2"
OUT_NAME="$MESH_NAME"
MESH_FILE="data/$MESH_NAME.xdmf"
TAGS_FILE="data/$MESH_NAME.pickle"

# 3. Create input-tmp.yml in current directory
INPUT_FILE="input.yml"
TMP_FILE="input-tmp.yml"
cp "$INPUT_FILE" "$TMP_FILE"

# 4. Override values in input-tmp.yml
# -- mesh_file
if grep -q "^[[:space:]]*mesh_file[[:space:]]*:" "$TMP_FILE"; then
  sed -i '' "s|^[[:space:]]*mesh_file[[:space:]]*:.*|mesh_file: \"$MESH_FILE\"|" "$TMP_FILE"
else
  echo "mesh_file: \"$MESH_FILE\"" >> "$TMP_FILE"
fi

# -- tags_dictionary_file
if grep -q "^[[:space:]]*tags_dictionary_file[[:space:]]*:" "$TMP_FILE"; then
  sed -i '' "s|^[[:space:]]*tags_dictionary_file[[:space:]]*:.*|tags_dictionary_file: \"$TAGS_FILE\"|" "$TMP_FILE"
else
  echo "tags_dictionary_file: \"$TAGS_FILE\"" >> "$TMP_FILE"
fi

# -- out_name
if grep -q "^[[:space:]]*out_name[[:space:]]*:" "$TMP_FILE"; then
  sed -i '' "s|^[[:space:]]*out_name[[:space:]]*:.*|out_name: \"$OUT_NAME\"|" "$TMP_FILE"
else
  echo "out_name: \"$OUT_NAME\"" >> "$TMP_FILE"
fi

# 5. Set output dir and copy input-tmp.yml
OUTPUT_DIR="$OUT_NAME"
mkdir -p "output/$OUTPUT_DIR"
cp "$TMP_FILE" "output/$OUTPUT_DIR/input.yml"

# 6. Run simulation
echo "Running simulation using: $TMP_FILE" | tee -a "$LOGFILE"

docker run -t -v "$(pwd)":/home/fenics -i ghcr.io/fenics/dolfinx/dolfinx:v0.9.0 \
  bash -c "
    cd /home/fenics && \
    pip install --no-build-isolation -r requirements.txt && \
    mpirun -n 5 python3 -u main.py input-tmp.yml
  " >> "$LOGFILE" 2>&1

# 7. Copy to external drive
DEST="/Volumes/IBT/cardioEMI-resolution/$OUT_NAME"
mkdir -p "$DEST"
cp -r "output/$OUTPUT_DIR"/* "$TMP_FILE" "$DEST/"
echo "Copied results to $DEST" | tee -a "$LOGFILE"

# 8. Cleanup
rm "$TMP_FILE"
rm -rf "output/$OUTPUT_DIR"
