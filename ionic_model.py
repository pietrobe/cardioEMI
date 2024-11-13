from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import dolfinx  as dfx
import numpy as np

class Ionic_model(ABC):

    # constructor
    def __init__(self, params):
        self.params = params                     
            
    @abstractmethod
    def _eval(self, phi_M):
        # Abstract method that must be implemented by concrete subclasses.
        pass

# Factory function
def ionic_model_factory(params, intra_intra=False):
    
    # Dictionary mapping strings to classes
    available_models = {
        "passive": Passive_model,
        "ohmic"  : Passive_model,
        "hodgkin–huxley": HH_model,
        "hh"            : HH_model,
        "ap"            : AP_model,
        "aliev-panfilov": AP_model
        # Add other models here as needed
    }
    
    model_name = params["ionic_model"]   

    if isinstance(model_name, str):

        model_name = model_name.lower()

        # Return an instance of the specified model
        if model_name in available_models:        
            return available_models[model_name](params)
        else:
            print("Available models: ", available_models)
            raise ValueError(f"Unknown model name: {model_name}")

    elif isinstance(model_name, dict):

        if intra_intra:
            model_name = model_name["intra_intra"].lower()

            if model_name in available_models:        
                return available_models[model_name](params)
            else:
                print("Available models: ", available_models)
                raise ValueError(f"Unknown model name: {model_name}")

        else:
            model_name = model_name["intra_extra"].lower()

            if model_name in available_models:        
                return available_models[model_name](params)
            else:
                print("Available models: ", available_models)
                raise ValueError(f"Unknown model name: {model_name}")
    else:
        raise ValueError(f"Unknown ionic model type")


# I_ch = phi_M / R_g
class Passive_model(Ionic_model):
    
    def __str__(self):
        return f'Passive'
        
    def _eval(self, phi_M):                 
        return (1.0/self.params["R_g"]) * phi_M


# Hodgkin–Huxley
class HH_model(Ionic_model):  

    # HH params
    # initial gating    
    n_init = 0.27622914792
    m_init = 0.03791834627
    h_init = 0.68848921811
    
    # conductivities
    g_Na_leak = 1          # Na leak conductivity (S/m**2)
    g_K_leak  = 4          # K leak conductivity (S/m**2)
    g_Cl_leak = 0          # Cl leak conductivity (S/m**2)        
    g_Na_bar  = 1200       # Na max conductivity (S/m**2)
    g_K_bar   = 360        # K max conductivity (S/m**2)        
    V_rest    = -0.065     # resting membrane potential
    E_Na      = 54.8e-3    # reversal potential Na (V)
    E_K       = -88.98e-3  # reversal potential K (V)
    E_Cl      = 0          # reversal potential 0 (V)
    
    # numerics
    use_Rush_Lar   = True
    time_steps_ODE = 25

    # save gating in PNG    
    save_png_file = False
    
    initial_time_step = True 
    

    def __str__(self):
        return f'Hodgkin–Huxley'
    

    def _eval(self, phi_M):   
        
        # update gating variables
        if self.initial_time_step:            
                                
            # create gating varaibles vectors 
            self.n = 0 * phi_M + self.n_init
            self.m = 0 * phi_M + self.m_init
            self.h = 0 * phi_M + self.h_init            

            # output
            if self.save_png_file: self.init_png()

            self.initial_time_step = False            

        else:            
            self.update_gating_variables(phi_M)  

            # output
            if self.save_png_file: self.save_png()                          

        # conductivities
        g_Na = self.g_Na_leak + self.g_Na_bar*self.m**3*self.h
        g_K  = self.g_K_leak  + self.g_K_bar *self.n**4              
        g_Cl = self.g_Cl_leak

        # ionic currents
        I_ch_Na = g_Na * (phi_M - self.E_Na)
        I_ch_K  = g_K  * (phi_M - self.E_K)
        I_ch_Cl = g_Cl * (phi_M - self.E_Cl)     
                
        return I_ch_Na + I_ch_K + I_ch_Cl      


    def update_gating_variables(self, phi_M):           

        dt_ode = float(self.params['dt'])/self.time_steps_ODE 

        V_M = 1000*(phi_M - self.V_rest) # convert phi_M to mV    
        
        alpha_n = 0.01e3*(10.-V_M)/(np.exp((10.-V_M)/10.) - 1.)
        beta_n  = 0.125e3*np.exp(-V_M/80.)
        alpha_m = 0.1e3*(25. - V_M)/(np.exp((25. - V_M)/10.) - 1)
        beta_m  = 4.e3*np.exp(-V_M/18.)
        alpha_h = 0.07e3*np.exp(-V_M/20.)
        beta_h  = 1.e3/(np.exp((30.-V_M)/10.) + 1)

        if self.use_Rush_Lar:
            
            tau_y_n = 1.0/(alpha_n + beta_n)
            tau_y_m = 1.0/(alpha_m + beta_m)
            tau_y_h = 1.0/(alpha_h + beta_h)

            y_inf_n = alpha_n * tau_y_n
            y_inf_m = alpha_m * tau_y_m
            y_inf_h = alpha_h * tau_y_h

            y_exp_n = np.exp(-dt_ode/tau_y_n)
            y_exp_m = np.exp(-dt_ode/tau_y_m)
            y_exp_h = np.exp(-dt_ode/tau_y_h)
            
        else:

            alpha_n *= dt_ode
            beta_n  *= dt_ode
            alpha_m *= dt_ode
            beta_m  *= dt_ode
            alpha_h *= dt_ode
            beta_h  *= dt_ode
        
        for i in range(self.time_steps_ODE): 

            if self.use_Rush_Lar:

                self.n = y_inf_n + (self.n - y_inf_n) * y_exp_n
                self.m = y_inf_m + (self.m - y_inf_m) * y_exp_m
                self.h = y_inf_h + (self.h - y_inf_h) * y_exp_h
                
            else:

                self.n += alpha_n * (1 - self.n) - beta_n * self.n
                self.m += alpha_m * (1 - self.m) - beta_m * self.m
                self.h += alpha_h * (1 - self.h) - beta_h * self.h 
        


# Aliev-Panfilov
class AP_model(Ionic_model):
    
    def __str__(self):
        return f'Aliev-Panfilov'
        
    def _eval(self, phi_M):                 

        # TODO
        I_ion = phi_M

        return I_ion






#   def init_png(self):

#       p = self.problem

#       self.point_to_plot = []     

#       # for gamma point
#       f_to_v = p.mesh.topology()(p.mesh.topology().dim()-1, 0)
#       dmap   = p.V.dofmap()           

#       # loop over facets updating gating only on gamma
#       for facet in facets(p.mesh):

#           if p.boundaries.array()[facet.index()] in p.gamma_tags:

#               vertices = f_to_v(facet.index())

#               local_indices = dmap.entity_closure_dofs(p.mesh, 0, [vertices[0]])              

#               self.point_to_plot = local_indices  

#               break                                           
                
#       # prepare data structures       
#       imap = dmap.index_map()
#       num_dofs_local = imap.size(IndexMap.MapSize.ALL) * imap.block_size()
        
#       local_n = self.n.vector().get_local(np.arange(num_dofs_local, dtype=np.int32))
#       local_m = self.m.vector().get_local(np.arange(num_dofs_local, dtype=np.int32))
#       local_h = self.h.vector().get_local(np.arange(num_dofs_local, dtype=np.int32))
        
#       self.n_t = []
#       self.m_t = []
#       self.h_t = []
        
#       self.n_t.append(local_n[self.point_to_plot]) 
#       self.m_t.append(local_m[self.point_to_plot]) 
#       self.h_t.append(local_h[self.point_to_plot]) 
        
#       self.out_gate_string = 'output/gating.png'
            

#   def save_png(self):
        
#       p = self.problem

#       # prepare data (needed for parallel)
#       dmap = p.V.dofmap()     
#       imap = dmap.index_map()
#       num_dofs_local = imap.size(IndexMap.MapSize.ALL) * imap.block_size()

#       local_n = self.n.vector().get_local(np.arange(num_dofs_local, dtype=np.int32))
#       local_m = self.m.vector().get_local(np.arange(num_dofs_local, dtype=np.int32))
#       local_h = self.h.vector().get_local(np.arange(num_dofs_local, dtype=np.int32))
                    
#       self.n_t.append(local_n[self.point_to_plot]) 
#       self.m_t.append(local_m[self.point_to_plot]) 
#       self.h_t.append(local_h[self.point_to_plot]) 


#   def plot_png(self):
        
#       # aliases
#       dt = float(self.problem.dt)
        
#       time_steps = len(self.n_t)

#       plt.figure(1)
#       plt.plot(np.linspace(0, 1000*time_steps*dt, time_steps), self.n_t, label='n')
#       plt.plot(np.linspace(0, 1000*time_steps*dt, time_steps), self.m_t, label='m')
#       plt.plot(np.linspace(0, 1000*time_steps*dt, time_steps), self.h_t, label='h')
#       plt.legend()
#       plt.xlabel('time (ms)')
#       plt.savefig(self.out_gate_string)