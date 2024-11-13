import dolfinx  as dfx
from petsc4py import PETSc
import scipy.sparse as sparse
import numpy        as np
from typing import Union
import numpy.typing as npt
from collections import defaultdict
import os
import yaml
import sys


def update_status(message):
    sys.stdout.write(f'\r{message}')
    sys.stdout.flush()


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



# Assign intial membrane potential
class Read_input_field:
    def __init__(self, expression: Union[str, float, int]):
        self.expression = expression

    def __call__(self, x: npt.NDArray) -> npt.NDArray:
        # If expression is a number, return it directly as an array of the same shape as `x`
        if isinstance(self.expression, (int, float)):
            return self.expression + 0 * x[0] # hack
        # If expression is a string, evaluate it
        elif isinstance(self.expression, str):
            return eval(self.expression, {"np": np, "x": x})
        else:
            raise ValueError("Expression must be a string, int, or float.")            


def read_input_file(input_yml_file):
        
        # read input yml file
        with open(input_yml_file, 'r') as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)        
            
        input_parameters = dict()

        ######### geometry #########
        if 'mesh_file' in config:              
            check_if_file_exists(config['mesh_file'])
            input_parameters['mesh_file'] = config['mesh_file']                                        
        else:
            print('INPUT ERROR: provide mesh_file field in input .yml file')
            return

        if 'tags_dictionary_file' in config:      
            check_if_file_exists(config['tags_dictionary_file'])
            input_parameters['tags_dictionary_file'] = config['tags_dictionary_file']                                        
        else:
            print('INPUT ERROR: provide tags_dictionary_file field in input .yml file')
            return
                
        ######### problem #########
        if 'dt' in config:
            input_parameters['dt'] = config['dt']
        else:
            print('INPUT ERROR: provide dt in input .yml file')
            return
        
        if 'time_steps' in config: 
            input_parameters['time_steps'] = config['time_steps']            
        elif 'T' in config:            
            input_parameters['time_steps'] = int(config['T']/config['dt'])        
        else:
            print('INPUT ERROR: provide final time T or time_steps in input .yml file.')
            exit()

        if 'mesh_conversion_factor' in config: 
            input_parameters['mesh_conversion_factor'] = config['mesh_conversion_factor']
        else:
            input_parameters['mesh_conversion_factor'] = 1.0
        
        # Membrane capacitance, (dafult 1) 
        if 'C_M' in config: 
            input_parameters['C_M'] = config['C_M']
        else:
            input_parameters['C_M'] = 1.0

        if 'sigma_i' in config: 
            input_parameters['sigma_i'] = config['sigma_i']
        else:
            input_parameters['sigma_i'] = 1.0

        if 'sigma_e' in config: 
            input_parameters['sigma_e'] = config['sigma_e']
        else:
            input_parameters['sigma_e'] = 1.0
                    
        # finite element polynomial order (dafult 1) 
        if 'fem_order' in config: 
            input_parameters['P'] = config['fem_order']
        else:
            input_parameters['P'] = 1
        
        # initial membrane potential (dafult 1)
        if 'phi_M_init' in config: 
            input_parameters['phi_M_init'] = config['phi_M_init']
        else:
            print('WARNING: initial membrane potential set to 1, set phi_M_init in input file for user defined one.')
            input_parameters['phi_M_init'] = "1"

        # ionic model 
        if 'ionic_model' in config: 
            input_parameters['ionic_model'] = config['ionic_model']
        else:
            print('WARNING: setting default passive ionic model')
            input_parameters['ionic_model'] = "Passive"

            
        ############### solver parameters ###############

        if 'ksp_type' in config: 
            input_parameters['ksp_type'] = config['ksp_type']
        else:
            input_parameters['ksp_type'] = 'cg'        

        if 'pc_type' in config: 
            input_parameters['pc_type'] = config['pc_type']
        else:
            input_parameters['pc_type'] = 'hypre'        

        if 'ksp_rtol' in config: 
            input_parameters['ksp_rtol'] = config['ksp_rtol']
        else:
            input_parameters['ksp_rtol'] = 1e-8
        
        if 'save_output' in config: 
            input_parameters['save_output'] = config['save_output']
        else:
            input_parameters['save_output'] = True

        if 'verbose' in config: 
            input_parameters['verbose'] = config['verbose']
        else:
            input_parameters['verbose'] = False
                
        return input_parameters
        

def check_if_file_exists(file_path):
    if not os.path.exists(file_path):        
        print(f"The file '{file_path}' does not exist.")
        exit()


def norm_2(vec):
    return sqrt(dot(vec,vec))


def dump(thing, path):
    if isinstance(thing, PETSc.Vec):
        assert np.all(np.isfinite(thing.array))
        return np.save(path, thing.array)
    m = sparse.csr_matrix(thing.getValuesCSR()[::-1]).tocoo()
    assert np.all(np.isfinite(m.data))
    return np.save(path, np.c_[m.row, m.col, m.data])


def common_elements(set1, set2):    
    return set1.intersection(set2)
