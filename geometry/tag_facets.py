import dolfinx  as dfx
from mpi4py      import MPI
from geometry    import *
import pickle 

mesh_folder = "../meshes_test/" 
basename    = mesh_folder + "test_physio_3D" #"2D_config5" #"test_physio_3D"
case_3d     = True
tagname     = "medit:ref" #"conductivity" #"medit:ref"

vtu_to_xdmf(basename, tagname, case_3d=case_3d)

# input mesh
mesh_file_volume_tagged = basename + ".xdmf" #"meshes_test/2D_config5.xdmf"

update_xdmf_name(mesh_file_volume_tagged)

# output mesh and dictionary
mesh_file_tagged       = basename + "_tagged.xdmf" #"meshes_test/2D_config5_tagged.xdmf"
connectivity_dict_file = basename + "_connectivity.pickle" #"meshes_test/2D_config5_connectivity.pickle"

with dfx.io.XDMFFile(MPI.COMM_WORLD, mesh_file_volume_tagged, 'r') as xdmf:
    # Read mesh and cell tags
    mesh       = xdmf.read_mesh(ghost_mode=dfx.mesh.GhostMode.shared_facet,name='mesh')    
    subdomains = xdmf.read_meshtags(mesh, name='mesh')        

##################################
# MESH PROCESSING HERE IF NEEDED #
##################################

N_TAGS = len(set(subdomains.values))

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