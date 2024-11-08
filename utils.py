#from dolfin import *
from petsc4py import PETSc
import scipy.sparse as sparse
import numpy        as np
from typing import Union
import numpy.typing as npt
import os
import yaml

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
