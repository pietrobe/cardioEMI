import dolfinx  as dfx
from mpi4py      import MPI
from collections import defaultdict
from geometry    import *
import pickle 

mesh_file     = "../data/mesh.xdmf"
out_mesh_file = "../data/reduced_tags_mesh.xdmf"
out_dict_file = "../data/reduced_tags_dict.pickle"

with dfx.io.XDMFFile(MPI.COMM_WORLD, mesh_file, 'r') as xdmf:
    # Read mesh and cell tags
    mesh       = xdmf.read_mesh(ghost_mode=dfx.mesh.GhostMode.shared_facet,name='mesh')    
    subdomains = xdmf.read_meshtags(mesh, name='mesh')        

num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
subdomain_values = subdomains.values 

# Loop over the cells in the mesh
for i in range(num_cells):
    
    update_status(f'Processing cells: {int(100*i/num_cells)}%')
    
    if subdomain_values[i] > 0:

        if subdomain_values[i] in [2, 3, 5, 7]:
            subdomain_values[i] = 2
        else:
            subdomain_values[i] = 1

N_TAGS = len(set(subdomain_values))

print("\nMarking facets", flush=True)
boundaries, membrane_tags_dict = get_facet_tags_and_dictionary(mesh, subdomains, N_TAGS)

print("\nSaving mesh with reduced tags", flush=True)
with dfx.io.XDMFFile(MPI.COMM_WORLD, out_mesh_file, "w") as mesh_file:
        mesh_file.write_mesh(mesh)
        boundaries.name = "facet_tags"
        subdomains.name = "cell_tags"
        mesh_file.write_meshtags(boundaries, mesh.geometry)
        mesh_file.write_meshtags(subdomains, mesh.geometry)

# Save to a file
with open(out_dict_file, "wb") as f:
    pickle.dump(membrane_tags_dict, f)   

