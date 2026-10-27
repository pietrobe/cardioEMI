#!/usr/bin/env bash
# Generate the Model A meshes of the paper (regular myocyte grids, Table 1):
#
#   A1  20 x 10 x 5    myocytes
#   A2  20 x 10 x 10
#   A3  20 x 20 x 10
#   A4  20 x 20 x 20
#
#   bash emigrid_mesh/generate_model_A.sh [A1 A2 A3 A4 | <name>=<nx>,<ny>,<nz> ...]
#
# Pipeline, for each mesh:
#   1. gmsh meshes emigrid.geo (one block per myocyte, gap-junction cubes on each
#      face, extracellular space around it; default lengths, in um);
#   2. convert_mesh.py writes it in openCARP format (.pts/.elem; odd tags = myocytes);
#   3. convert_pts_elem.py maps the extracellular space to tag 0, colours the
#      myocytes so that neighbours never share a tag (3 tags in total) and writes
#      <name>.xdmf/.h5 with cell and facet tags plus the <name>.pickle connectivity
#      dictionary read by main.py.
# Outputs go to $CARDIO_MESH_DIR (default: meshes/ at the repository root), where
# the benchmark inputs expect them.
#
# Requirements: gmsh >= 4.11 (GMSH, default `gmsh`) and a Python (PYTHON, default
# `python3`) with numpy, meshio, h5py and lxml, e.g.
#   python3 -m venv mesh-env && mesh-env/bin/pip install gmsh numpy meshio h5py lxml
#   GMSH=mesh-env/bin/gmsh PYTHON=mesh-env/bin/python bash emigrid_mesh/generate_model_A.sh A1
# Element counts may differ slightly between gmsh versions (gmsh 4.13.1 gives
# about 8,291 tetrahedra per myocyte, as in Table 1).
# Cost: the conversion is serial Python; A4 (66M tetrahedra) needs a large-memory
# node and several hours. Intermediate files are removed unless KEEP_INTERMEDIATE=1.

set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${CARDIO_MESH_DIR:-$HERE/../meshes}
GMSH=${GMSH:-gmsh}
PYTHON=${PYTHON:-python3}
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)

grid_of() {
    case $1 in A1) echo 20,10,5 ;; A2) echo 20,10,10 ;; A3) echo 20,20,10 ;; A4) echo 20,20,20 ;; esac
}

for arg in "${@:-A1 A2 A3 A4}"; do
    for spec in $arg; do
        if [[ $spec == *=* ]]; then name=${spec%%=*}; grid=${spec#*=}
        else name=$spec; grid=$(grid_of "$spec"); fi
        [ -n "$grid" ] || { echo "unknown mesh $spec (A1-A4 or <name>=<nx>,<ny>,<nz>)" >&2; exit 2; }
        IFS=, read -r nx ny nz <<< "$grid"
        echo ">>> $name: $nx x $ny x $nz myocytes -> $OUT/$name.{xdmf,h5,pickle} ($(date '+%F %T'))"
        base=$OUT/$name
        "$GMSH" -3 "$HERE/emigrid.geo" -setnumber nx "$nx" -setnumber ny "$ny" -setnumber nz "$nz" \
            -o "$base.msh" > "$base.gmsh.log"
        tail -n 2 "$base.gmsh.log"
        "$PYTHON" "$HERE/convert_mesh.py" "$base.msh"
        "$PYTHON" "$HERE/convert_pts_elem.py" "$base.pts" "$base.elem" "$base" --color-intracellular
        if [ "${KEEP_INTERMEDIATE:-0}" != 1 ]; then
            rm -f "$base.msh" "$base.pts" "$base.elem" "$base.intra" "$base.extra" "$base.gmsh.log"
        fi
        echo ">>> $name done ($(date '+%F %T'))"
    done
done
