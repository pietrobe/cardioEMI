"""
This script imports synthetic meshes that were converted with meshio from vtu to xdmf
Note that, in these meshes, there is some unnecessary and some missing data to be used for cardioEMI
Here, we drop the unnecessary data, rename the keys according to what cardioEMI can read and generate the 
missing data.
"""

import h5py
import numpy as np
from lxml import etree
from collections import defaultdict
import pickle
import subprocess
import os
import glob

def convert_mesh(input_file, output_file):
    try:
        # Run the meshio conversion command
        subprocess.run(
            ["meshio", "convert", input_file, output_file],
            check=True  # Raise an exception if the command fails
        )
        print(f"Successfully converted {input_file} to {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during mesh conversion: {e}")
    except FileNotFoundError:
        print("meshio is not installed or not in the system PATH.")

def load_xdmf_and_h5(xdmf_file, h5_file):
    # Parse the XDMF file
    xdmf_tree = etree.parse(xdmf_file)
    root = xdmf_tree.getroot()

    # Open the HDF5 file
    with h5py.File(h5_file, "r") as h5:
        # Function to safely retrieve data from HDF5
        def get_data_from_hdf5(path):
            try:
                return np.array(h5[path])
            except KeyError:
                raise KeyError(f"Path '{path}' not found in HDF5 file.")

        # Extract geometry data
        geometry_item = root.xpath("//Grid[@Name='Grid']/Geometry/DataItem")[0]
        geometry_path = geometry_item.text.strip().split(":")[-1]
        geometry_data = get_data_from_hdf5(geometry_path)

        # Extract topology data
        topology_item = root.xpath("//Grid[@Name='Grid']/Topology/DataItem")[0]
        topology_path = topology_item.text.strip().split(":")[-1]
        topology_data = get_data_from_hdf5(topology_path)

        # Extract node-based attribute
        node_attribute_item = root.xpath("//Grid[@Name='Grid']/Attribute[@Center='Node']/DataItem")[0]
        node_attribute_path = node_attribute_item.text.strip().split(":")[-1]
        node_attribute_data = get_data_from_hdf5(node_attribute_path)

        # Extract cell-based attribute
        cell_attribute_item = root.xpath("//Grid[@Name='Grid']/Attribute[@Center='Cell']/DataItem")[0]
        cell_attribute_path = cell_attribute_item.text.strip().split(":")[-1]
        cell_attribute_data = get_data_from_hdf5(cell_attribute_path)

    # Return the data as a dictionary
    return {
        "geometry": geometry_data,
        "topology": topology_data,
        "node_attribute": node_attribute_data,
        "cell_attribute": cell_attribute_data,
    }

def reduce_tags(mesh):
    geometry = mesh['geometry']
    topology = mesh['topology']
    cell_values = mesh['cell_values']

    # Step 1: Create adjacency for tetrahedra
    def find_neighbors(topology):
        face_to_cells = defaultdict(list)
        for cell_id, tetra in enumerate(topology):
            faces = [tuple(sorted(tetra[[i, j, k]])) for i, j, k in [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]]
            for face in faces:
                face_to_cells[face].append(cell_id)
        adjacency = defaultdict(set)
        for cells in face_to_cells.values():
            if len(cells) == 2:
                c1, c2 = cells
                adjacency[c1].add(c2)
                adjacency[c2].add(c1)
        return adjacency

    adjacency = find_neighbors(topology)

    # Step 2: Build conflict graph
    tag_neighbors = defaultdict(set)
    for cell, neighbors in adjacency.items():
        cell_tag = cell_values[cell]
        for neighbor in neighbors:
            neighbor_tag = cell_values[neighbor]
            if cell_tag != neighbor_tag:  # Only consider different tags
                tag_neighbors[cell_tag].add(neighbor_tag)
                tag_neighbors[neighbor_tag].add(cell_tag)

    # Step 3: Graph coloring (greedy)
    tag_colors = {}
    for tag in sorted(tag_neighbors):  # Sort tags for deterministic behavior
        neighbor_colors = {tag_colors[neighbor] for neighbor in tag_neighbors[tag] if neighbor in tag_colors}
        tag_colors[tag] = next(color for color in range(len(tag_neighbors)) if color not in neighbor_colors)

    # Step 4: Map new colors to cells
    new_cell_values = np.array([tag_colors[tag] for tag in cell_values])

    return {
        'geometry': geometry,
        'topology': topology,
        'cell_values': new_cell_values
    }

def facets(mesh):
    # Step 1: Duplicate cell topology
    mesh['cell_topology'] = mesh['topology'].copy()
    
    # Step 2: Extract facets from tetrahedra
    def extract_facets(topology):
        facets = defaultdict(list)
        for cell_id, tetra in enumerate(topology):
            # Each tetrahedron contributes 4 facets
            face_combinations = [tuple(sorted(tetra[[i, j, k]])) for i, j, k in [
                (1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]]
            for face in face_combinations:
                facets[face].append(cell_id)
        return facets

    facets = extract_facets(mesh['topology'])

    # Step 3: Assign facet values
    facet_topology = []
    facet_values = []
    zero_combination_map = {}
    combination_map = {}
    boundary_map = defaultdict(set)  # To store the result
    next_tag = 2  # Start unique tags from 2

    for facet, cells in facets.items():
        facet_topology.append(facet)
        if len(cells) == 1:  # Boundary facet
            cell_value = mesh['cell_values'][cells[0]]
            if cell_value == 0:
                facet_tag = 0
            else:
                if (0, cell_value) not in zero_combination_map:
                    zero_combination_map[(0, cell_value)] = next_tag
                    next_tag += 1
                facet_tag = zero_combination_map[(0, cell_value)]
            facet_values.append(facet_tag)
            if facet_tag!=0:
                boundary_map[cell_value].add(facet_tag)  # Add to boundary map
        elif len(cells) == 2:  # Interior facet
            cell_value_1 = mesh['cell_values'][cells[0]]
            cell_value_2 = mesh['cell_values'][cells[1]]
            if cell_value_1 == cell_value_2:
                facet_tag = 0  # (n, n)
            elif 0 in (cell_value_1, cell_value_2):
                n = max(cell_value_1, cell_value_2)  # Get the non-zero value
                if (0, n) not in zero_combination_map:
                    zero_combination_map[(0, n)] = next_tag
                    next_tag += 1
                facet_tag = zero_combination_map[(0, n)]
            else:
                # Assign unique tags for (n, m) where n != m, n, m > 0
                n, m = sorted((cell_value_1, cell_value_2))  # Ensure (min, max) order
                if (n, m) not in combination_map:
                    combination_map[(n, m)] = next_tag
                    next_tag += 1
                facet_tag = combination_map[(n, m)]
            facet_values.append(facet_tag)
            # Add to boundary map for both cell_value_1 and cell_value_2
            if facet_tag!=0:
                boundary_map[cell_value_1].add(facet_tag)
                boundary_map[cell_value_2].add(facet_tag)

    # Step 4: Add to mesh
    mesh['facet_topology'] = np.array(facet_topology)
    mesh['facet_values'] = np.array(facet_values)

    return mesh, zero_combination_map, combination_map, boundary_map

def write_xdmf_h5(mesh, xdmf_file, h5_file):
    # Namespace for xi
    xi_ns = "https://www.w3.org/2001/XInclude"
    nsmap = {"xi": xi_ns}

    # Write the HDF5 file
    with h5py.File(h5_file, "w") as h5:
        # Create groups for mesh data
        mesh_group = h5.create_group("Mesh")
        mesh_group.create_dataset("mesh/geometry", data=mesh['geometry'])
        mesh_group.create_dataset("mesh/topology", data=mesh['topology'])

        # Create groups for facet tags
        mesh_group.create_dataset("facet_tags/topology", data=mesh['facet_topology'])
        mesh_group.create_dataset("facet_tags/Values", data=mesh['facet_values'].ravel())

        # Create groups for cell tags
        mesh_group.create_dataset("cell_tags/topology", data=mesh['cell_topology'])
        mesh_group.create_dataset("cell_tags/Values", data=mesh['cell_values'].ravel())

    # Write the XDMF file
    root = etree.Element("Xdmf", Version="3.0", nsmap=nsmap)
    domain = etree.SubElement(root, "Domain")

    # Mesh Grid
    grid_mesh = etree.SubElement(domain, "Grid", Name="mesh", GridType="Uniform")
    topology_mesh = etree.SubElement(grid_mesh, "Topology", TopologyType="Tetrahedron",
                                      NumberOfElements=str(mesh['topology'].shape[0]))
    etree.SubElement(topology_mesh, "DataItem", Dimensions=f"{mesh['topology'].shape[0]} 4", 
                     NumberType="Int", Format="HDF").text = f"{h5_file}:/Mesh/mesh/topology"

    geometry_mesh = etree.SubElement(grid_mesh, "Geometry", GeometryType="XYZ")
    etree.SubElement(geometry_mesh, "DataItem", Dimensions=f"{mesh['geometry'].shape[0]} 3",
                     Format="HDF").text = f"{h5_file}:/Mesh/mesh/geometry"

    # Facet Tags Grid
    grid_facet = etree.SubElement(domain, "Grid", Name="facet_tags", GridType="Uniform")
    xi_include_geom = etree.Element(f"{{{xi_ns}}}include", 
                                     xpointer="xpointer(/Xdmf/Domain/Grid/Geometry)")
    grid_facet.append(xi_include_geom)
    topology_facet = etree.SubElement(grid_facet, "Topology", TopologyType="Triangle",
                                       NumberOfElements=str(mesh['facet_topology'].shape[0]))
    etree.SubElement(topology_facet, "DataItem", Dimensions=f"{mesh['facet_topology'].shape[0]} 3", 
                     NumberType="Int", Format="HDF").text = f"{h5_file}:/Mesh/facet_tags/topology"

    attribute_facet = etree.SubElement(grid_facet, "Attribute", Name="facet_tags", 
                                       AttributeType="Scalar", Center="Cell")
    etree.SubElement(attribute_facet, "DataItem", Dimensions=f"{mesh['facet_values'].shape[0]}",
                     Format="HDF").text = f"{h5_file}:/Mesh/facet_tags/Values"

    # Cell Tags Grid
    grid_cell = etree.SubElement(domain, "Grid", Name="cell_tags", GridType="Uniform")
    xi_include_geom = etree.Element(f"{{{xi_ns}}}include", 
                                     xpointer="xpointer(/Xdmf/Domain/Grid/Geometry)")
    grid_cell.append(xi_include_geom)
    topology_cell = etree.SubElement(grid_cell, "Topology", TopologyType="Tetrahedron",
                                      NumberOfElements=str(mesh['cell_topology'].shape[0]))
    etree.SubElement(topology_cell, "DataItem", Dimensions=f"{mesh['cell_topology'].shape[0]} 4", 
                     NumberType="Int", Format="HDF").text = f"{h5_file}:/Mesh/cell_tags/topology"

    attribute_cell = etree.SubElement(grid_cell, "Attribute", Name="cell_tags", 
                                       AttributeType="Scalar", Center="Cell")
    etree.SubElement(attribute_cell, "DataItem", Dimensions=f"{mesh['cell_values'].shape[0]}",
                     Format="HDF").text = f"{h5_file}:/Mesh/cell_tags/Values"

    # Save the XDMF file
    tree = etree.ElementTree(root)
    tree.write(xdmf_file, pretty_print=True, xml_declaration=True, encoding="UTF-8")

def clear():
    # Get the current working directory
    current_dir = os.getcwd()
    
    # Find all files matching the pattern "tmp.*"
    tmp_files = glob.glob(os.path.join(current_dir, "tmp.*"))
    
    # Remove each file
    for tmp_file in tmp_files:
        try:
            os.remove(tmp_file)
            print(f"Removed: {tmp_file}")
        except Exception as e:
            print(f"Failed to remove {tmp_file}: {e}")

def main():
    folder = "../meshes_test/"
    input_mesh = folder + 'robin.vtu'
    prefix_out = folder + 'robin'

    # Step 1: Convert synthetic mesh generator output to .xdmf
    convert_mesh(input_mesh, 'tmp.xdmf')

    # Step 2: Load mesh converted via meshio
    mesh = load_xdmf_and_h5('tmp.xdmf', 'tmp.h5')

    # Step 3: Rename and delete unnecessary keys
    if 'node_attribute' in mesh:
        del mesh['node_attribute']
    if 'cell_attribute' in mesh:
        mesh['cell_values'] = mesh.pop('cell_attribute')

    # Step 4: Minimise tag numbers ensuring that no neighbouring subdomains share the same tag
    mesh = reduce_tags(mesh)

    # Step 5: Create facet data and duplicate topology values into a new key called cell_topology
    # Facet tags are 0 for inside a domain, and unique tags for intersections of domains (including I-E)
    mesh, zero_combinations, unique_combinations, boundary_map = facets(mesh)

    # Step 6: Write pickle file (dictionary for membrane tags)
    with open(f"{prefix_out}.pickle", "wb") as f:
        pickle.dump(boundary_map, f)

    # Step 7: Write final mesh to XDMF and HDF5 formats
    write_xdmf_h5(mesh, f"{prefix_out}.xdmf", f"{prefix_out}.h5")

    # Step 8: Clean up temporary files
    clear()

if __name__ == "__main__":
    main()