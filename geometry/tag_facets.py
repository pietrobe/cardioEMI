import dolfinx  as dfx
from mpi4py      import MPI
from collections import defaultdict
from geometry    import *
import pickle 

# input mesh
mesh_file_volume_tagged = "../data/mesh.xdmf"

# output mesh and dictionary
mesh_file_tagged       = "../data/tagged_mesh.xdmf"
connectivity_dict_file = "../data/connectivity_dict.pickle"

with dfx.io.XDMFFile(MPI.COMM_WORLD, mesh_file_volume_tagged, 'r') as xdmf:
    # Read mesh and cell tags
    mesh       = xdmf.read_mesh(ghost_mode=dfx.mesh.GhostMode.shared_facet,name='mesh')    
    subdomains = xdmf.read_meshtags(mesh, name='mesh')        

##################################
# MESH PROCESSING HERE IF NEEDED #
##################################

N_TAGS = len(set(subdomain_values))

print("\nMarking facets", flush=True)
boundaries, membrane_tags_dict = get_facet_tags_and_dictionary(mesh, subdomains, N_TAGS)

print("\nSaving mesh with reduced tags", flush=True)
with dfx.io.XDMFFile(MPI.COMM_WORLD, mesh_file_tagged, "w") as mesh_file:
        mesh_file.write_mesh(mesh)
        boundaries.name = "facet_tags"
        subdomains.name = "cell_tags"
        mesh_file.write_meshtags(boundaries, mesh.geometry)
        mesh_file.write_meshtags(subdomains, mesh.geometry)

# Save to a file
with open(connectivity_dict_file, "wb") as f:
    pickle.dump(membrane_tags_dict, f)   

