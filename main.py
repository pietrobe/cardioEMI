import ufl
import time
import pickle 
import multiphenicsx.fem
import multiphenicsx.fem.petsc 
import dolfinx  as dfx
from ufl      import inner, grad
from sys      import argv
from mpi4py   import MPI
from pathlib  import Path
from petsc4py import PETSc
from utils             import *
from ionic_model       import *

    # Options for the fenicsx form compiler optimization
cache_dir       = f"{str(Path.cwd())}/.cache"
compile_options = ["-Ofast","-march=native"]
jit_parameters  = {"cffi_extra_compile_args"  : compile_options,
                    "cache_dir"               : cache_dir,
                    "cffi_libraries"          : ["m"]}

#----------------------------------------#
#     PARAMETERS AND SOLVER SETTINGS     #
#----------------------------------------#

start_time = time.perf_counter()

# MPI communicator
comm = MPI.COMM_WORLD 

# Read input file 
if comm.rank == 0: print("Processing input file:", argv[1])      
params = read_input_file(argv[1])

# aliases
mesh_file = params["mesh_file"]
ECS_TAG   = params["ECS_TAG"]
dt        = params["dt"]

# get expression of initial mmebrane potential
v_init = Read_input_field(params['phi_M_init'])

# Timers
solve_time    = 0
assemble_time = 0

#-----------------------#
#          MESH         #
#-----------------------#

if comm.rank == 0: print("Input mesh file:", mesh_file)       

with open(params["tags_dictionary_file"], "rb") as f:
    membrane_tags = pickle.load(f)

# set tags info
TAGS   = sorted(membrane_tags.keys())
N_TAGS = len(TAGS)

# Read mesh
with dfx.io.XDMFFile(MPI.COMM_WORLD, mesh_file, 'r') as xdmf:
    # Read mesh and cell tags
    mesh       = xdmf.read_mesh(ghost_mode=dfx.mesh.GhostMode.shared_facet)
    subdomains = xdmf.read_meshtags(mesh, name="cell_tags")    

    # Create facet-to-cell connectivity
    mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)

    # Also the identity is needed
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)

    boundaries = xdmf.read_meshtags(mesh, name="facet_tags")

# Scale mesh
mesh.geometry.x[:] *= params["mesh_conversion_factor"]

# timers
if comm.rank == 0: print(f"Reading input time: {time.perf_counter() - start_time:.2f}")
t1 = time.perf_counter()

# Define integral measures
dx = ufl.Measure("dx", subdomain_data=subdomains) # Cell integrals
dS = ufl.Measure("dS", subdomain_data=boundaries) # Facet integrals

# Read physical constants
sigma_i = read_input_field(params['sigma_i'], mesh=mesh)
sigma_e = read_input_field(params['sigma_e'], mesh=mesh)
tau     = dt/params["C_M"]

#------------------------------------------#
#     FUNCTION SPACES AND RESTRICTIONS     #
#------------------------------------------#
V = dfx.fem.functionspace(mesh, ("Lagrange", params["P"])) # Space for functions defined on the entire mesh

# vector for space, one for each tag
V_dict = dict()

# trial and test functions
u_dict = dict()
v_dict = dict()

# list for storing the solutions and forcing factors
uh_dict  = dict()
vij_dict = dict()
fg_dict  = dict()

# to store all solutions
v = dfx.fem.Function(V)

for i in TAGS:

    V_i = V.clone()

    V_dict[i]  = V_i
    u_dict[i]  = ufl.TrialFunction(V_i)
    v_dict[i]  =  ufl.TestFunction(V_i)
    uh_dict[i] =  dfx.fem.Function(V_i)
    
    # v_ij con i < j to avoid repetions
    for j in TAGS:
        if i < j:
            # Membrane potential and forcing term function
            vij_dict[(i,j)] = dfx.fem.Function(V) 
            fg_dict[(i,j)]  = dfx.fem.Function(V)
        
# init vij using initial membrane potential        
for i in TAGS:    

        # interpolate v_init in intra_extra, intra_intra is 0 by default
    if i < ECS_TAG:    
        vij_dict[(i,ECS_TAG)].interpolate(v_init)    
        v.x.array[:] += vij_dict[(i,ECS_TAG)].x.array[:] 

    elif i > ECS_TAG:
        vij_dict[(ECS_TAG,i)].interpolate(v_init)    
        v.x.array[:] += vij_dict[(ECS_TAG,i)].x.array[:] 

##### Restrictions #####
restriction = []

for i in TAGS:

    V_i = V_dict[i]

    # Get indices of the cells of the intra- and extracellular subdomains
    cells_Omega_i = subdomains.indices[subdomains.values == i]

    # Get dofs of the intra- and extracellular subdomains
    dofs_Vi_Omega_i = dfx.fem.locate_dofs_topological(V_i, subdomains.dim, cells_Omega_i)
    
    # Define the restrictions of the subdomains
    restriction_Vi_Omega_i = multiphenicsx.fem.DofMapRestriction(V_i.dofmap, dofs_Vi_Omega_i)

    restriction.append(restriction_Vi_Omega_i)

# timers
if comm.rank == 0: print(f"Creating FEM spaces and restrictions: {time.perf_counter() - t1:.2f}")
t1 = time.perf_counter()

# set ionic models
ionic_models = dict()

for i in TAGS:        
    for j in TAGS:

        if i < j:        
            if i == ECS_TAG or j == ECS_TAG:
                ionic_models[(i,j)] = ionic_model_factory(params, intra_intra=False)
            else:
                ionic_models[(i,j)] = ionic_model_factory(params, intra_intra=True)
        
#------------------------------------#
#        VARIATIONAL PROBLEM         #
#------------------------------------#

# BCs (use bcs=[bc_point] in block_assembly)
zero     = dfx.fem.Constant(mesh, PETSc.ScalarType(0.0))
bc_dofs  = dfx.fem.locate_dofs_topological(V_dict[ECS_TAG], boundaries.dim, [ECS_TAG])
bc_point = dfx.fem.dirichletbc(zero, bc_dofs, V_dict[ECS_TAG]) 

# bilinear form
a = []

# assemble block form
for i in TAGS:

    a_i = []

    # membranes tags for cell tag i 
    membrane_i = membrane_tags[i]

    if i == ECS_TAG:
        sigma = sigma_e # extra-cellular 

    else:
        sigma = sigma_i # intra-cellular         

    v_i = v_dict[i]
    
    for j in TAGS:
        
        u_j  = u_dict[j]

        membrane_ij = tuple(common_elements(membrane_i,membrane_tags[j]))   

        # if cells i and j have a membrane in common
        if len(membrane_ij) > 0:                 

            if i == j:                                
                a_ij = tau * inner(sigma * grad(u_j), grad(v_i)) * dx(i) + inner(u_j('-'), v_i('-')) * dS(membrane_ij)
            else:
                a_ij = - inner(u_j('+'), v_i('-')) * dS(membrane_ij)            
        else:
            a_ij = None

        a_i.append(a_ij)   

    a.append(a_i)

# Convert form to dolfinx form
a = dfx.fem.form(a, jit_options=jit_parameters)

# timers
if comm.rank == 0: print(f"Creating bilinear form: {time.perf_counter() - t1:.2f}")
t1 = time.perf_counter() 

#---------------------------#
#      MATRIX ASSEMBLY      #
#---------------------------#

# Assemble the block linear system matrix
A = multiphenicsx.fem.petsc.assemble_matrix_block(a, restriction=(restriction, restriction), bcs=[bc_point])
A.assemble()
assemble_time += time.perf_counter() - t1 # Add time lapsed to total assembly time

if comm.rank == 0: print(f"Assembling matrix A: {time.perf_counter() - t1:.2f}")

# Save A
# dump(A, 'output/Amat')

# Configure solver
ksp = PETSc.KSP()
ksp.create(comm)
ksp.setOperators(A)

# Set solver
ksp.setType(params["ksp_type"])
ksp.getPC().setType(params["pc_type"])
if params['pc_type'] == "lu":
    ksp.getPC().setFactorSolverType("mumps")
opts = PETSc.Options()

if params["verbose"]:
    opts.setValue('ksp_view', None)
    opts.setValue('ksp_monitor_true_residual', None)

# for titerastive solvers set tolerance 
if params['pc_type'] != "lu" and params['ksp_type'] != "preonly":    
    opts.setValue('ksp_rtol', params["ksp_rtol"])
    opts.setValue('ksp_converged_reason', None)

if params['pc_type'] == "hypre" and mesh.geometry.dim == 3:
    opts.setValue('pc_hypre_boomeramg_strong_threshold', 0.7)

ksp.setFromOptions()

# intial time
t = 0.0

# Create output files
if params["save_output"]:

    # rename solutions
    for i in TAGS:        
        uh_dict[i].name  = "u_" + str(i)
    
    # potentials xdmf
    out_sol = dfx.io.XDMFFile(comm, "output/solution.xdmf", "w")
    out_sol.write_mesh(mesh)            
    # out_sol.write_meshtags(subdomains, mesh.geometry) # TODO remove??
    
    # memebrane potential xdmf
    out_v = dfx.io.XDMFFile(comm, "output/v.xdmf" , "w")
    out_v.write_mesh(mesh)
    out_v.write_function(v, t)

    # save subdomain data, needed for parallel visualizaiton
    with dfx.io.XDMFFile(comm, "output/tags.xdmf", "w") as out_tags:                
        out_tags.write_mesh(mesh)            
        out_tags.write_meshtags(subdomains, mesh.geometry)
        out_tags.write_meshtags(boundaries, mesh.geometry)        
        out_tags.close()

#---------------------------------#
#        SOLUTION TIMELOOP        #
#---------------------------------#

# init auxiliary data structures
ksp_iterations = []
I_ion = dict()

for time_step in range(params["time_steps"]):

    if comm.rank == 0: print("time_step =", time_step )

    # init data structure for linear form
    L_list = []

    # Increment time
    t += dt

    # Update and assemble vector that is the RHS of the linear system
    t1 = time.perf_counter() # Timestamp for assembly time-lapse      
    
    for i in TAGS:

        membrane_i = membrane_tags[i]
        
        v_i = v_dict[i]

        L_i = 0    

        for j in TAGS:                        
            
            if i != j:
            
                membrane_ij = tuple(common_elements(membrane_i,membrane_tags[j]))   
                
                if i < j:
                    ij_tuple = (i,j)                                        
                    L_coeff  = 1
                    with vij_dict[ij_tuple].vector.localForm() as v_local:
                        I_ion[ij_tuple] = ionic_models[ij_tuple]._eval(v_local[:])                                            
                else:
                    ij_tuple = (j,i)
                    L_coeff  = -1                    
                    
                with fg_dict[ij_tuple].vector.localForm() as fg_local, vij_dict[ij_tuple].vector.localForm() as v_local:

                    fg_local[:] = v_local[:] - tau * I_ion[ij_tuple]

                L_i += L_coeff * inner(fg_dict[ij_tuple], v_i('+')) * dS(membrane_ij)
                                
        L_list.append(L_i)

    L = dfx.fem.form(L_list, jit_options=jit_parameters) # Convert form to dolfinx form

    b = multiphenicsx.fem.petsc.assemble_vector_block(L, a, restriction=restriction, bcs=[bc_point]) # Assemble RHS vector
    
    # dump(b, 'output/bvec')
        
    if time_step == 0:
        # Create solution vector
        sol_vec = multiphenicsx.fem.petsc.create_vector_block(L, restriction=restriction)

    assemble_time += time.perf_counter() - t1 # Add time lapsed to total assembly time
    
    # Solve the system
    t1 = time.perf_counter() # Timestamp for solver time-lapse
    ksp.solve(b, sol_vec)
        
    # store iterisons 
    ksp_iterations.append(ksp.getIterationNumber())

    # Update ghost values
    sol_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    
    # Extract sub-components of solution
    dofmap_list = (N_TAGS) * [V.dofmap]
    with multiphenicsx.fem.petsc.BlockVecSubVectorWrapper(sol_vec, dofmap_list, restriction) as uij_wrapper:
        for ui_ue_wrapper_local, component in zip(uij_wrapper, tuple(uh_dict.values())): 
            with component.vector.localForm() as component_local:
                component_local[:] = ui_ue_wrapper_local

    for i in TAGS:
        for j in TAGS:
            if i < j:                
                vij_dict[(i,j)].x.array[:] = uh_dict[i].x.array - uh_dict[j].x.array

    # reset v
    with v.vector.localForm() as v_local:
        v_local.set(0)

    for i in TAGS:
        v.x.array[:] += uh_dict[i].x.array[:]     

    solve_time += time.perf_counter() - t1 # Add time lapsed to total solver time

    # save xdmf output
    if params["save_output"] and time_step % params["save_interval"] == 0:               
        for i in TAGS:
            out_sol.write_function(uh_dict[i], t)        

        out_v.write_function(v, t)

#------------------------------#
#         POST PROCESS         #
#------------------------------#
# Sum local assembly and solve times to get global values
max_local_assemble_time = comm.allreduce(assemble_time, op=MPI.MAX) # Global assembly time
max_local_solve_time    = comm.allreduce(solve_time   , op=MPI.MAX) # Global solve time

# Print stuff
if comm.rank == 0: 
    print("\n#-----------INFO-----------#\n")
    print("MPI size =", comm.size)        
    print("N_TAGS ="  , N_TAGS   )
    print("dt ="      , dt       )
    print("P ="       , params["P"])
    print("ksp_type =", params["ksp_type"])
    print("pc_type =" , params["pc_type"] )
    print("Average KSP iterations =", sum(ksp_iterations)/len(ksp_iterations))
    print("Global #DoFs =", b.getSize())
    
    if isinstance(params["ionic_model"], dict):
        print("Ionic models:")
        for key, value in params["ionic_model"].items():
            print(f"  {key}: {value}")
    else:
        print("Ionic model:", params['ionic_model'])        


    print("\n#-------TIME ELAPSED-------#\n")
    print(f"Assembly time:      {max_local_assemble_time:.3f} seconds")
    print(f"Solve time:         {max_local_solve_time:.3f}    seconds")
    print(f"Total time elapsed: {time.perf_counter() - start_time:.3f} seconds")

# Write solutions to file
if params["save_output"]:    
    if comm.rank == 0: print("\nSolution saved in output")

    out_sol.close()
    out_v.close()