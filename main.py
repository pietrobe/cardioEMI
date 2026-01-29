import ufl
import time
import pickle
try:
    import multiphenicsx.fem
    import multiphenicsx.fem.petsc
except ImportError:
    print("Missing multiphenicsx! Code will only run with CUDA enabled.")
import dolfinx  as dfx
import json
import numpy as np
from ufl      import inner, grad
from sys      import argv, stdout
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

# Timers
solve_time    = 0
assemble_time = 0
ODEs_time     = 0

start_time = time.perf_counter()

# MPI communicator
comm = MPI.COMM_WORLD

if comm.rank == 0:
    print("\n#-----------SETUP----------#")
    print("Processing input file:", argv[1])

# Read input file
params = read_input_file(argv[1])

# aliases
mesh_file = params["mesh_file"]
ECS_TAG   = params["ECS_TAG"]
dt        = params["dt"]
cuda = params["cuda"]

if cuda:
    import cudolfinx as cufem

#-----------------------#
#          MESH         #
#-----------------------#

if comm.rank == 0: print("Input mesh file:", mesh_file)

with open(params["tags_dictionary_file"], "rb") as f:
    membrane_tags = pickle.load(f)

# set tags info
TAGS   = sorted(membrane_tags.keys())
N_TAGS = len(TAGS)

if comm.rank == 0: print("Num tags", TAGS)

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
    if cuda and comm.size > 1:
        mesh = cufem.ghost_layer_mesh(mesh)
        subdomains = cufem.ghost_layer_meshtags(subdomains, mesh)
        boundaries = cufem.ghost_layer_meshtags(boundaries, mesh)
        # Recreate connectivities as we have a new mesh
        mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
        mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)

# Scale mesh
mesh.geometry.x[:] *= params["mesh_conversion_factor"]

# timers
if comm.rank == 0: print(f"Reading input time:     {time.perf_counter() - start_time:.2f} seconds")
t1 = time.perf_counter()

# Define integral measures
dx = ufl.Measure("dx", subdomain_data=subdomains) # Cell integrals
dS = ufl.Measure("dS", subdomain_data=boundaries) # Facet integrals

# Read physical constants
sigma_i = read_input_field(params['sigma_i'], mesh=mesh)
sigma_e = read_input_field(params['sigma_e'], mesh=mesh)
tau     = dt/params["C_M"]

# add electrode conductivity if present
using_electrode = ("sigma_electrode" in params)

if using_electrode:    
    sigma_electrode = read_input_field(params['sigma_electrode'], mesh=mesh)
    ELECTRODE_TAG   = params["ELECTRODE_TAG"]

#------------------------------------------#
#     FUNCTION SPACES AND RESTRICTIONS     #
#------------------------------------------#
V = dfx.fem.functionspace(mesh, ("Lagrange", params["P"])) # Space for functions defined on the entire mesh

# trial and test functions
u_dict = dict()
v_dict = dict()

# list for storing the solutions and forcing factors
uh_dict  = dict()
vij_dict = dict()
fg_dict  = dict()

# to store membrane potential
v = dfx.fem.Function(V)
v.name = "v"

for i in TAGS:

    u_dict[i]  = ufl.TrialFunction(V)
    v_dict[i]  =  ufl.TestFunction(V)
    uh_dict[i] =  dfx.fem.Function(V)
    
    # v_ij con i < j to avoid repetions
    for j in TAGS:
        if i < j:
            # Membrane potential and forcing term function
            vij_dict[(i,j)] = dfx.fem.Function(V)
            fg_dict[(i,j)]  = dfx.fem.Function(V)

# get expression of initial membrane potential
v_init_expr = read_input_field(params['v_init'], V=V)

# turn expression into a Function with actual DOF values
v_init = dfx.fem.Function(V)
v_init.interpolate(v_init_expr)
        
# init vij using initial membrane potential
for i in TAGS:

    # interpolate v_init in intra_extra, intra_intra is 0 by default
    if i < ECS_TAG:
        vij_dict[(i,ECS_TAG)].interpolate(v_init)
        # v.x.array[:] += vij_dict[(i,ECS_TAG)].x.array[:]

    elif i > ECS_TAG:
        vij_dict[(ECS_TAG,i)].interpolate(v_init)

# save membrane potential for visualization (valid only for extra-intra)
v.x.array[:] = vij_dict[(TAGS[0],TAGS[1])].x.array[:]

##### Restrictions #####
restriction = []
restriction_dof_list = []
restriction_local_sizes = []
tot_dofs = 0
for i in TAGS:

    V_i = V

    # Get indices of the cells of the intra- and extracellular subdomains
    cells_Omega_i = subdomains.indices[subdomains.values == i]

    if i == ECS_TAG and using_electrode:        
        cells_Omega_electrode = subdomains.indices[subdomains.values == ELECTRODE_TAG]      
        cells_Omega_i = np.concatenate([cells_Omega_i, cells_Omega_electrode])                          

    # Get dofs of the intra- and extracellular subdomains
    dofs_Vi_Omega_i = dfx.fem.locate_dofs_topological(V_i, subdomains.dim, cells_Omega_i)
    tot_dofs += len(dofs_Vi_Omega_i)

    # Define the restrictions of the subdomains
    if cuda:
        restriction_dof_list.append(dofs_Vi_Omega_i)
        local_size = int(sum(dofs_Vi_Omega_i<V_i.dofmap.index_map.size_local))
        restriction_local_sizes.append(local_size)
    else:
        restriction_Vi_Omega_i = multiphenicsx.fem.DofMapRestriction(V_i.dofmap, dofs_Vi_Omega_i)
        restriction.append(restriction_Vi_Omega_i)
#if comm.rank == 0: print("Sum of dofs across tags", tot_dofs, " total ", V.dofmap.index_map.size_global)
# timers
if comm.rank == 0: print(f"Creating FEM spaces:    {time.perf_counter() - t1:.2f} seconds")
t1 = time.perf_counter()
setup_time = t1 - start_time

# set ionic models
ionic_models = dict()

for i in TAGS:
    for j in TAGS:

        if i < j:
            if i == ECS_TAG or j == ECS_TAG:
                ionic_models[(i,j)] = ionic_model_factory(params, intra_intra=False)
            else:
                ionic_models[(i,j)] = ionic_model_factory(params, intra_intra=True, V=V)

if comm.rank == 0: print(f"making ionic models: {time.perf_counter() - t1:.2f} seconds")        

####### BC #######
number_of_Dirichlet_points = params['Dirichlet_points']
Dirichletbc = (number_of_Dirichlet_points > 0) 

bcs = []

if Dirichletbc:

    # Apply zero Dirichlet condition
    zero = dfx.fem.Constant(mesh, 0.0)        

    # identify local boundary DOFs + coords
    boundary_facets = dfx.mesh.exterior_facet_indices(mesh.topology)
    local_bdofs = dfx.fem.locate_dofs_topological(V, mesh.topology.dim-1, boundary_facets)
    coords = V.tabulate_dof_coordinates()
    local_coords = coords[local_bdofs]      # shape (n_loc, gdim)

    # local to global
    imap = V.dofmap.index_map
    first_global = imap.local_range[0]       # first global index on this rank
    local_global_bdofs = first_global + local_bdofs

    # gather everyone’s cands to rank 0
    all_globals = comm.gather(local_global_bdofs, root=0)
    all_coords  = comm.gather(local_coords,      root=0)

    # on rank 0 pick the 10 “corner‐nearest” by taxi‐distance
    if comm.rank == 0:
        G  = np.concatenate(all_globals)
        C  = np.vstack(all_coords)
        scores = C.sum(axis=1)
        chosen_global = G[np.argsort(scores)[:number_of_Dirichlet_points]]
    else:
        chosen_global = None

    # broadcast the final 10 GLOBAL DOFs to everyone
    chosen_global = comm.bcast(chosen_global, root=0)

    # each rank picks from its local globals, maps back to local indices
    mask = np.isin(local_global_bdofs, chosen_global)
    local_chosen = local_bdofs[mask].astype(np.int32)

    # impose BCs only on these local_chosen
    for i in TAGS:
        bc_i = dfx.fem.dirichletbc(zero, local_chosen, V)
        bcs.append(bc_i)

##############
#------------------------------------#
#        VARIATIONAL PROBLEM         #
#------------------------------------#

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

                if i == ECS_TAG and using_electrode:
                    a_ij +=  tau * inner(sigma_electrode * grad(u_j), grad(v_i)) * dx(ELECTRODE_TAG)                       

            else:                                
                a_ij = - inner(u_j('+'), v_i('-')) * dS(membrane_ij)                                      
        else:
            a_ij = None

        a_i.append(a_ij)   

    a.append(a_i)
if cuda:
    asm = cufem.CUDAAssembler()
    if comm.rank == 0: print("Making forms.")
    cuda_a = cufem.form(a, restriction=(restriction_dof_list, restriction_dof_list))
    if comm.rank == 0: print("calling create matrix block.")
    cuda_A = asm.create_matrix_block(cuda_a)
else:
    # Converte form to dolfinx form
    a = dfx.fem.form(a, jit_options=jit_parameters)

# timers
if comm.rank == 0: print(f"Creating bilinear form: {time.perf_counter() - t1:.2f} seconds")
t1 = time.perf_counter() 

# #---------------------------#
# #      MATRIX ASSEMBLY      #
# #---------------------------#

if cuda:
  asm.assemble_matrix_block(cuda_a, cuda_A)
  cuda_A.assemble()
  A = cuda_A.mat
else:
  # Assemble the block linear system matrix
  A = multiphenicsx.fem.petsc.assemble_matrix_block(a, bcs=bcs, restriction=(restriction, restriction))
  A.assemble()
print(f"A norm {A.norm()}")
assemble_time += time.perf_counter() - t1 # Add time lapsed to total assembly time
matrix_assemble_time = assemble_time

if comm.rank == 0: print(f"Assembling matrix A:    {time.perf_counter() - t1:.2f} seconds")

#---------------------------------#
#        CREATE NULLSPACE         #
#---------------------------------#

if not Dirichletbc:
    # Create the PETSc nullspace vector and check that it is a valid nullspace of A
    nullspace = PETSc.NullSpace().create(constant=True,comm=comm)
    assert nullspace.test(A)
    # For convenience, we explicitly inform PETSc that A is symmetric, so that it automatically
    # sets the nullspace of A^T too (see the documentation of MatSetNullSpace).
    # Symmetry checked also by direct inspection through the plot_sparsity_pattern() function
    A.setOption(PETSc.Mat.Option.SYMMETRIC, True)
    A.setOption(PETSc.Mat.Option.SYMMETRY_ETERNAL, True)
    # Set the nullspace
    A.setNullSpace(nullspace)

#---------------------------------#
#      CONFIGURE SOLVER           #
#---------------------------------#

if cuda:
    A.setType('aijcusparse')

# Configure solver
ksp = PETSc.KSP().create(comm)
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

# for iterative solvers set tolerance
if params['pc_type'] != "lu" and params['ksp_type'] != "preonly":
    opts.setValue('ksp_rtol', params["ksp_rtol"])
    opts.setValue('ksp_converged_reason', None)

if params['pc_type'] == "hypre":
    opts.setValue("pc_hypre_boomeramg_relax_type_all", "l1scaled-SOR/Jacobi")
    opts.setValue("pc_hypre_boomeramg_agg_nl", 1)
    if mesh.geometry.dim == 3:
        opts.setValue('pc_hypre_boomeramg_strong_threshold', 0.25)
opts.setValue("ksp_norm_type", "unpreconditioned")
if "petsc_opts" in params:
    for k,v in params["petsc_opts"].items():
        if comm.rank == 0: print(f"Setting solver option '{k}' to '{v}'")
        opts.setValue(k,v)
ksp.setFromOptions()

# intial time
t = 0.0

# Create output files
if params["save_output"]:

    # rename solutions
    for i in TAGS:        
        uh_dict[i].name  = "u_" + str(i)
    
    out_name = params.get("out_name", "").strip().lstrip("_")

    # potentials xdmf
    out_sol = dfx.io.XDMFFile(comm, out_name + "/solution.xdmf", "w")
    out_sol.write_mesh(mesh)            
        
    # memebrane potential xdmf
    out_v = dfx.io.XDMFFile(comm, out_name + "/v.xdmf" , "w")
    out_v.write_mesh(mesh)
    out_v.write_function(v, t)

    # save subdomain data, needed for parallel visualizaiton
    with dfx.io.XDMFFile(comm, out_name + "/tags.xdmf", "w") as out_tags:                     
        out_tags.write_mesh(mesh)            
        out_tags.write_meshtags(subdomains, mesh.geometry)
        out_tags.write_meshtags(boundaries, mesh.geometry)        
        out_tags.close()


#---------------------------------#
#        STIMULUS SETUP           #
#---------------------------------#

# user parameters
stim_expr  = params.get("I_stim", "100.0 * (x[0] < 0.03)")
stim_start = params.get("stim_start", 0.0)  # ms
stim_end   = params.get("stim_end", 1.0)    # ms

# Build a stimulus Function per tag/space (so spaces match v_i and uh_dict[i])
stim_fun = {}
coords = V.tabulate_dof_coordinates().reshape((-1, mesh.geometry.dim))
xlist = [coords[:, 0], coords[:, 1], coords[:, 2]]
vals  = eval(stim_expr, {"x": xlist, "np": np})
f = dfx.fem.Function(V)
f.x.array[:] = np.asarray(vals, dtype=float)
for i in TAGS:
    stim_fun[i] = f

# Time-dependent amplitude as a Constant (UFL-safe)
stim_amp = dfx.fem.Constant(mesh, PETSc.ScalarType(0.0))

ksp_iterations = []
I_ion = {}

#---------------------------------#
#        SOLUTION TIMELOOP        #
#---------------------------------#

# init auxiliary data structures
ksp_iterations = []
#I_ion = dict()

if comm.rank == 0: print("\n#-----------SOLVE----------#")

failed = False

for time_step in range(params["time_steps"]):

    if comm.rank == 0: update_status(f'Time stepping: {int(100*time_step/params["time_steps"])}%')

    # physical time at current step (before advancing)
    t_n = float(time_step) * float(dt)

    # update stimulus amplitude based on current time
    if (stim_start <= t_n) and (t_n < stim_end):
        stim_amp.value = 1.0
    else:
        stim_amp.value = 0.0

    # init data structure for linear form
    L_list = []

    # Update and assemble vector that is the RHS of the linear system
    t1 = time.perf_counter() # Timestamp for assembly time-lapse      
    
    for i in TAGS:

        membrane_i = membrane_tags[i]
        
        v_i = v_dict[i]

        L_i = 0    

        for j in TAGS:                        
            
            if i != j:
                if time_step == 0: 
                    membrane_ij = tuple(common_elements(membrane_i,membrane_tags[j]))   
                
                if i < j:
                    ij_tuple = (i,j)                                        
                    L_coeff  = 1
                    with vij_dict[ij_tuple].x.petsc_vec.localForm() as v_local:

                        t_ODE = time.perf_counter()
                        
                        I_ion[ij_tuple] = ionic_models[ij_tuple]._eval(v_local[:])          

                        ODEs_time += time.perf_counter() - t_ODE 
                else:
                    ij_tuple = (j,i)
                    L_coeff  = -1                    
                with fg_dict[ij_tuple].x.petsc_vec.localForm() as fg_local, vij_dict[ij_tuple].x.petsc_vec.localForm() as v_local:

                    fg_local[:] = v_local[:] - tau * I_ion[ij_tuple]

                if time_step == 0:
                    L_i += L_coeff * inner(fg_dict[ij_tuple], v_i('+')) * dS(membrane_ij)

                    # external stimulus (time-switched by Constant)
                    if ECS_TAG in (i, j):
                        L_i += L_coeff * tau * stim_amp * inner(stim_fun[i], v_i('+')) * dS(membrane_ij)

        if time_step == 0:                        
            L_list.append(L_i)

    # Increment time
    t += float(dt)

    t_test = time.perf_counter()
    
    # create some data structures
    if time_step == 0: 
        if cuda:
            L = cufem.form(L_list, restriction=restriction_dof_list)
            cuda_b = asm.create_vector_block(L)
            b = cuda_b.vector
            sol_vec = b.copy()
        else:
            # Convert form to dolfinx form                    
            L = dfx.fem.form(L_list, jit_options=jit_parameters) 

            # Create right-hand side and solution vectors        
            b       = multiphenicsx.fem.petsc.create_vector_block(L, restriction=restriction)
            sol_vec = multiphenicsx.fem.petsc.create_vector_block(L, restriction=restriction)
        vector_assemble_setup_time = time.perf_counter() - t_test
    if cuda:
        asm.assemble_vector_block(L, cuda_b)
    else:
        # Clear RHS vector to avoid accumulation and assemble RHS
        b.array[:] = 0
        b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        multiphenicsx.fem.petsc.assemble_vector_block(b, L, a, bcs=bcs, restriction=restriction) # Assemble RHS vector        

    if not Dirichletbc:
        # if the timestep is not zero, b changes anyway and the nullspace must be removed
        nullspace.remove(b)
    print(f"b norm {b.norm()}")
    assemble_time += time.perf_counter() - t1 # Add time lapsed to total assembly time

    
    # Solve the system
    t1 = time.perf_counter() # Timestamp for solver time-lapse
    ksp.solve(b, sol_vec)
    # store iterisons 
    ksp_iterations.append(ksp.getIterationNumber())
    if ksp.getConvergedReason() < 0:
        print("failed for ", ksp.getConvergedReason())
        failed = True
        break
    if not cuda:
        # Update ghost values
        sol_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    print(f"sol_vec norm {sol_vec.norm()}")
    # Extract sub-components of solution
    if cuda:
        offset = 0
        for i, restriction_dofs, size in zip(TAGS, restriction_dof_list, restriction_local_sizes):
            uh_dict[i].x.array[restriction_dofs[:size]] = sol_vec.array[offset:offset+size]
            uh_dict[i].x.scatter_forward()
            offset += size

    else:
        dofmap_list = (N_TAGS) * [V.dofmap]
        with multiphenicsx.fem.petsc.BlockVecSubVectorWrapper(sol_vec, dofmap_list, restriction) as uij_wrapper:
            for ui_ue_wrapper_local, component in zip(uij_wrapper, tuple(uh_dict.values())): 
                with component.x.petsc_vec.localForm() as component_local:
                    component_local[:] = ui_ue_wrapper_local

    for i in TAGS:
        for j in TAGS:
            if i < j:                
                vij_dict[(i,j)].x.array[:] = uh_dict[i].x.array - uh_dict[j].x.array # TODO test other order?
                
    
    solve_time += time.perf_counter() - t1 # Add time lapsed to total solver time

    # save xdmf output
    if params["save_output"] and time_step % params["save_interval"] == 0:               
        for i in TAGS:
            out_sol.write_function(uh_dict[i], t)        
        # fill v for visualization
        v.x.array[:] = uh_dict[ECS_TAG].x.array

        for i in TAGS:
            if i != ECS_TAG:
                v.x.array[:] -= uh_dict[i].x.array

        out_v.write_function(v, t)


if comm.rank == 0: update_status(f'Time stepping: 100%')        

#------------------------------#
#         POST PROCESS         #
#------------------------------#
# Sum local assembly and solve times to get global values
max_local_assemble_time = comm.allreduce(assemble_time, op=MPI.MAX) # Global assembly time
max_local_solve_time    = comm.allreduce(solve_time   , op=MPI.MAX) # Global solve time
max_local_ODE_time      = comm.allreduce(ODEs_time    , op=MPI.MAX) # Global ODEs time
max_local_setup_time    = comm.allreduce(setup_time   , op=MPI.MAX) # Global setup time
max_local_matrix_assemble_time = comm.allreduce(matrix_assemble_time, op=MPI.MAX)
max_local_vector_assemble_time = comm.allreduce(assemble_time-matrix_assemble_time, op=MPI.MAX)
max_local_vector_assemble_setup_time = comm.allreduce(vector_assemble_setup_time, op=MPI.MAX)
total_time = max_local_assemble_time + max_local_solve_time + max_local_ODE_time + max_local_setup_time

# Print stuff
if comm.rank == 0: 
    num_dofs = b.getSize()
    avg_ksp_its = sum(ksp_iterations)/len(ksp_iterations)
    print("\n\n#-----------INFO-----------#")
    print("MPI size     =", comm.size)        
    print("N_TAGS       =", N_TAGS   )
    print("dt           =", dt       )
    print("time steps   =", params["time_steps"])
    print("T            =", dt * params["time_steps"])
    print("P (FE order) =", params["P"])
    print("ksp_type     =", params["ksp_type"])
    print("pc_type      =", params["pc_type"] )
    print("Global #DoFs =", num_dofs)
    print("Average KSP iterations =", avg_ksp_its)
    if failed:
        print("LINEAR SOLVER FAILED!!!")
    
    if isinstance(params["ionic_model"], dict):
        print("Ionic models:")
        for key, value in params["ionic_model"].items():
            print(f"  {key}: {value}")
    else:
        print("Ionic model:", params['ionic_model'])        


    print("\n#-------TIME ELAPSED-------#")
    print(f"Setup time:       {max_local_setup_time:.3f} seconds")
    print(f"Assembly time:    {max_local_assemble_time:.3f} seconds")
    print(f"Solve time:       {max_local_solve_time:.3f} seconds")
    print(f"Ionic model time: {max_local_ODE_time:.3f} seconds")
    print(f"Total time:       {total_time:.3f} seconds")    
    

if params["save_output"]:    

    out_sol.close()
    out_v.close()

    if comm.rank == 0: 
        print("\nSolution saved in output folder")    
        print(f"Total script time (with output): {time.perf_counter() - start_time:.3f} seconds\n")

if params["save_performance"] and comm.rank == 0:
    stats = {
        "setup": max_local_setup_time,
        "assemble": max_local_assemble_time,
        "matrix_assemble": max_local_matrix_assemble_time,
        "vector_assemble_setup": max_local_vector_assemble_setup_time,
        "vector_assemble": max_local_vector_assemble_time,
        "solve": max_local_solve_time,
        "ionic": max_local_ODE_time,
        "total": total_time,
        "num_dofs": num_dofs,
        "avg_ksp_its": avg_ksp_its,
        "failed": failed
    }
    with open(params["out_name"]+f"-stats-{'cuda' if cuda else 'cpu'}-{comm.size}.json", "w") as fp:
        json.dump({"input": params, "performance": stats}, fp)
