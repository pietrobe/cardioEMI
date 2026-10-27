#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [ "meshio" ]
# ///
"Convert msh to openCARP format."

import meshio
import numpy as np
import sys
import os

if __name__ == "__main__":
    
    # FIXME: add binary format and vtu output
    if len(sys.argv) < 2:
        print("Missing mesh filename argument.")
        print(f"Usage: {sys.argv[0]} emigrid.msh")
        sys.exit(1)

    mesh_filename = sys.argv[1]
    base, ext = os.path.splitext(mesh_filename)

    # outputs
    pts_fn   = f"{base}.pts"
    elem_fn  = f"{base}.elem"
    intra_fn = f"{base}.intra"
    extra_fn = f"{base}.extra"
    
    msh = meshio.read(mesh_filename)
    pts = msh.points
    elm = msh.cells_dict['tetra']
    tag = np.hstack(msh.cell_data['gmsh:physical'])
    ntags = len(msh.cell_data['gmsh:physical'])

    print(f"Mesh with {pts.shape[0]} vertices, {elm.shape[0]} tetra, {ntags} tags")

    with open(pts_fn, "w") as fo:
        print(pts.shape[0], file=fo)
        np.savetxt(fo, pts)

    arr = np.empty((elm.shape[0], 6), dtype=object)
    arr[:, 0] = "Tr"
    arr[:, 1:5] = elm[:, 0:4]
    arr[:, 5] = tag

    intra = np.arange(ntags)[1::2]
    extra = np.arange(ntags)[2::2]

    with open(elem_fn, "w") as fo:
        print(arr.shape[0], file=fo)
        np.savetxt(fo, arr, fmt='%s %d %d %d %d %d')

    with open(intra_fn, 'w') as fo:
        print(intra.shape[0], file=fo)
        np.savetxt(fo, intra, fmt='%d')

    with open(extra_fn, 'w') as fo:
        print(extra.shape[0], file=fo)
        np.savetxt(fo, extra, fmt='%d')

