import numpy.typing
import numpy    as np
import dolfinx  as dfx
from sys         import argv
from mpi4py      import MPI
from collections import defaultdict
import json
import sys
import time
import pickle 

# in 2D this works
N_TAGS = 4 

# user input
elements_per_side = 256 
cells_per_side    = 32
out_file = "../data/square_mesh_" + str(elements_per_side) + "_" + str(cells_per_side) + ".xdmf"
out_dict = "../data/square_dict_" + str(elements_per_side) + "_" + str(cells_per_side) + ".pickle"

t1 = time.perf_counter()


def update_status(message):
    sys.stdout.write(f'\r{message}')
    sys.stdout.flush()

def mark_subdomains_square_many_cells(mesh: dfx.mesh.Mesh, cells_per_side: int) -> dfx.mesh.MeshTags:

    # if cells_per_side > 64:
    #     print("WARNING: cells with same tag could share facet. Try increasing N_TAGS.")

    # geometry parameters
    XY_MIN = 0.125
    XY_MAX = 0.875
    DIGIT_TOL = 10

    # Tag values
    EXTRA = 0

    cell_length = (XY_MAX - XY_MIN) / cells_per_side
    
    def create_nth_inside_function(n):

        x_min = round(XY_MIN + cell_length * (n % cells_per_side),  DIGIT_TOL)
        y_min = round(XY_MIN + cell_length * (n // cells_per_side), DIGIT_TOL)

        x_max = round(x_min + cell_length, DIGIT_TOL)
        y_max = round(y_min + cell_length, DIGIT_TOL)
        

        def inside(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
            """ Locator function for the inner square. """

            bool1 = np.logical_and(x[0] <= x_max, x[0] >= x_min) # True if inside inner box in x range
            bool2 = np.logical_and(x[1] <= y_max, x[1] >= y_min) # True if inside inner box in y range
        
            return np.logical_and(bool1, bool2)

        return inside


    cell_dim = mesh.topology.dim
    
    # Generate mesh topology
    mesh.topology.create_entities(cell_dim)
    mesh.topology.create_connectivity(cell_dim, cell_dim - 1)
    
    # Get total number of cells and set default facet marker value to OUTER
    num_cells   = mesh.topology.index_map(cell_dim).size_local + mesh.topology.index_map(cell_dim).num_ghosts
    cell_marker = np.full(num_cells, EXTRA, dtype = np.int32)

    total_intra_cells = cells_per_side * cells_per_side

    # Get all facets
    for i in range(total_intra_cells):

        row_index = i % cells_per_side
        even_row  = (i // cells_per_side) % 2

        # print("---------")
        # print("even_row =", even_row)
        # print("row_index =", row_index)
        # print("i =",i)
        # print("tag =", 1 + (2 * even_row + row_index) % N_TAGS)

        inside_fun = create_nth_inside_function(i)

        inner_cells = dfx.mesh.locate_entities(mesh, cell_dim, inside_fun)
        cell_marker[inner_cells] = 1 + (2 * even_row + row_index) % N_TAGS
        #cell_marker[inner_cells] = 1 + (even_row + row_index) % N_TAGS        # uncomment and set N_TAGS = 2 for the chequered tag

    cell_tags = dfx.mesh.meshtags(mesh, cell_dim, np.arange(num_cells, dtype = np.int32), cell_marker)

    return cell_tags

def mark_boundaries_square_many_cells(mesh: dfx.mesh.Mesh, cells_per_side: int, subdomains: dfx.mesh.MeshTags) -> dfx.mesh.MeshTags:    

    DEFAULT = -5
    
    cell_dim  = mesh.topology.dim
    facet_dim = cell_dim - 1

    # Generate mesh topology
    mesh.topology.create_connectivity(facet_dim, cell_dim)

    # map
    facet_to_cell = mesh.topology.connectivity(facet_dim, cell_dim)    

    # Get total number of facets
    num_facets = mesh.topology.index_map(facet_dim).size_local + mesh.topology.index_map(facet_dim).num_ghosts

    # init facet tag array
    facet_marker = np.full(num_facets, DEFAULT, dtype = np.int32)
    
    # given a subdomain tag, returns all the corresponding membrane tag
    membrane_tags_dict = defaultdict(set)

    # test
    timer_1 = 0
    timer_2 = 0
    
    # loop over facets and set facet_tags
    for facet_index in range(num_facets):
        
        # Get the cells connected to the facet
        connected_cells       = facet_to_cell.links(facet_index)        
        connected_cells_tags  = []

        update_status(f'Processing facets: {int(100*facet_index/num_facets)}%')

        if len(connected_cells) == 2:                    

            for cell_index in connected_cells:
                
                tag_index = np.where(subdomains.indices == cell_index)[0]                       
                cell_tag  = subdomains.values[tag_index[0]]                                            
                connected_cells_tags.append(cell_tag)
            
            if connected_cells_tags[0] != connected_cells_tags[1]:             

                membrane_tag  = min(connected_cells_tags) * (N_TAGS + 1) + max(connected_cells_tags)
                facet_marker[facet_index] = membrane_tag

                membrane_tags_dict[connected_cells_tags[0]].add(membrane_tag) 
                membrane_tags_dict[connected_cells_tags[1]].add(membrane_tag)                             
    
    facet_tags = dfx.mesh.meshtags(mesh, facet_dim, np.arange(num_facets, dtype = np.int32), facet_marker)

    return membrane_tags_dict, facet_tags

#-----------------------#
#          MESH         #
#-----------------------#
def create_square(N: int, cells_per_side: int, out_file: str) -> dict:

    print("Creating mesh " + out_file)
    
    comm       = MPI.COMM_SELF # MPI communicator
    ghost_mode = dfx.mesh.GhostMode.shared_facet # How dofs are distributed in parallel

    if comm.size > 1 and comm.rank == 0:
        print("ERROR: mesh creation only in serial")
        exit()

    # Create mesh
    mesh = dfx.mesh.create_unit_square(comm, N, N, cell_type=dfx.mesh.CellType.triangle, ghost_mode=ghost_mode)

    # Get subdomains and boundaries
    print("Marking cells", flush=True)
    subdomains = mark_subdomains_square_many_cells(mesh, cells_per_side)

    print("Marking facets", flush=True)
    membrane_tags_dict, boundaries  = mark_boundaries_square_many_cells(mesh, cells_per_side, subdomains)

    print("\nSaving mesh", flush=True)
    with dfx.io.XDMFFile(mesh.comm, out_file, "w") as mesh_file:
            mesh_file.write_mesh(mesh)
            boundaries.name = "facet_tags"
            subdomains.name = "cell_tags"
            mesh_file.write_meshtags(boundaries, mesh.geometry)
            mesh_file.write_meshtags(subdomains, mesh.geometry)

    # Save to a file
    with open(out_dict, "wb") as f:
        pickle.dump(membrane_tags_dict, f)   

    return membrane_tags_dict


create_square(N=elements_per_side, cells_per_side=cells_per_side, out_file=out_file)
print(f"Create mesh time: {time.perf_counter()-t1:.2f}")