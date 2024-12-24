import os
import dolfinx as dfx
from mpi4py import MPI

# refine tagged mesh (connectivity dictionary stays the same)

# set here the mesh name to be refined (with no extension)
input_xdmf  = "../data/square_mesh_64_1.xdmf" # CHANGE HERE

if not os.path.exists(input_xdmf):        
    print(f"The file '{file_path}' does not exist.")
    exit()

output_xdmf = input_xdmf.removesuffix(".xdmf") + "_refined.xdmf"

# Load the mesh from the input XDMF file
with dfx.io.XDMFFile(MPI.COMM_WORLD, input_xdmf, 'r') as xdmf:
  
    # Read mesh and cell tags
    mesh = xdmf.read_mesh()

    subdomains = xdmf.read_meshtags(mesh, name="cell_tags")        

    # Create facet-to-cell and cell-cell connectivities
    mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)

    boundaries = xdmf.read_meshtags(mesh, name="facet_tags")


# Perform mesh refinement
mesh_fine, parent_cells, parent_facets = dfx.mesh.refine_plaza(mesh, False, dfx.mesh.RefinementOption.parent_cell_and_facet)
mesh_fine.topology.create_connectivity(mesh_fine.topology.dim - 1, mesh_fine.topology.dim)

subdomains_fine = dfx.mesh.transfer_meshtag(subdomains, mesh_fine, parent_cells)
boundaries_fine = dfx.mesh.transfer_meshtag(boundaries, mesh_fine, parent_cells, parent_facets)

# Save the refined mesh to an output XDMF file
with dfx.io.XDMFFile(MPI.COMM_WORLD, output_xdmf, "w") as xdmf:
    xdmf.write_mesh(mesh_fine)
    
    boundaries_fine.name = "facet_tags"
    subdomains_fine.name = "cell_tags"

    xdmf.write_meshtags(subdomains_fine, mesh_fine.geometry)
    xdmf.write_meshtags(boundaries_fine, mesh_fine.geometry)      
    xdmf.close()  
        

print(f"Mesh refinement completed. Refined mesh saved to {output_xdmf}.")
