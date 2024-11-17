import numpy    as np
import dolfinx  as dfx
import meshio
from collections import defaultdict
import sys
import xml.etree.ElementTree as ET

def update_xdmf_name(filename):
    # Step 1: Parse the XML
    tree = ET.parse(filename)
    root = tree.getroot()

    # Step 2: Find the Grid element and modify its name attribute
    for grid in root.iter("Grid"):
        if grid.attrib.get("Name") == "Grid":  # Check if the name is "Grid"
            grid.attrib["Name"] = "mesh"  # Update to "mesh"

    # Step 3: Save the updated XML back to the file
    tree.write(filename, encoding="UTF-8", xml_declaration=True)

    print("Grid name updated")

def update_status(message):
    sys.stdout.write(f'\r{message}')
    sys.stdout.flush()

def clean_2D_mesh(mesh):
    # Step 2: Identify duplicate nodes
    points = mesh.points
    unique_points, inverse_indices = np.unique(points.round(decimals=12), axis=0, return_inverse=True)

    # Step 3: Update connectivity
    # Replace old node indices with unique node indices
    new_connectivity = []
    for cell_block in mesh.cells:
        if cell_block.type == "triangle":  # Process triangle cells
            new_cell_data = inverse_indices[cell_block.data]
            new_connectivity.append(new_cell_data)

    return unique_points, new_connectivity[0]

def vtu_to_xdmf(basename, tagname, case_3d=False):
    vtu_file = basename + ".vtu"
    msh = meshio.read(vtu_file)

    for cell in msh.cells:
        cells = cell.data
    
    for key in msh.cell_data_dict[tagname].keys():
        mesh_data = msh.cell_data_dict[tagname][key]

    if case_3d:
        points = msh.points
        converted_mesh = meshio.Mesh(points=points, cells={"tetra": cells}, cell_data={"cell_tags" : [mesh_data]})
    else:
        points, cells = clean_2D_mesh(msh)
        converted_mesh = meshio.Mesh(points=points, cells={"triangle": cells}, cell_data={"cell_tags" : [mesh_data]})

    meshio.write(basename + ".xdmf", converted_mesh)

def get_facet_tags_and_dictionary(mesh: dfx.mesh.Mesh, subdomains: dfx.mesh.MeshTags, N_TAGS: int) -> dfx.mesh.MeshTags:    

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

    return facet_tags, membrane_tags_dict
