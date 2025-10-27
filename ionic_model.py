from abc import ABC, abstractmethod
#import matplotlib.pyplot as plt
from utils import *
import dolfinx  as dfx
import numpy as np
import math

class Ionic_model(ABC):

    # constructor
    def __init__(self, params, V=None):
        self.params = params                     
        self.V      = V

    @abstractmethod
    def _eval(self, v):
        # Abstract method that must be implemented by concrete subclasses.
        pass

# Factory function
def ionic_model_factory(params, intra_intra=False, V=None):
    
    # Dictionary mapping strings to classes
    available_models = {
        "null"   : Null_model,
        "passive": Passive_model,
        "ohmic"  : Passive_model,
        "hodgkin–huxley": HH_model,
        "hh"            : HH_model,
        "ap"            : AP_model,
        "aliev-panfilov": AP_model,
        "ap2"           : AP_model_2,
        "courtemanche"  : Courtemanche_model,
        # Add other models here as needed
    }
    
    model_name = params["ionic_model"]   

    if isinstance(model_name, str):

        model_name = model_name.lower()

        # Return an instance of the specified model
        if (model_name in available_models) and (V is not None):        
            return available_models[model_name](params, V)
        elif (model_name in available_models) and (V is None):
            return available_models[model_name](params)
        else:
            print("Available models: ", available_models)
            raise ValueError(f"Unknown model name: {model_name}")

    elif isinstance(model_name, dict):

        if intra_intra:
            model_name = model_name["intra_intra"].lower()

            if model_name=="passive" or model_name=="ohmic":  
                return available_models[model_name](params,V)
            elif model_name in available_models:        
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


# I_ch = 0
class Null_model(Ionic_model):    
    def __str__(self):
        return f'Null'
        
    def _eval(self, v):              
        return 0 * v


# I_ch = v / R_g
class Passive_model(Ionic_model):
    def __init__(self, params, V):        
        super().__init__(params, V=V)
        #R_g = Read_input_field(self.params["R_g"])
        #self.R_g = dfx.fem.Function(V)
        #self.R_g.interpolate(R_g)
        R_g_expr = read_input_field(self.params["R_g"], V=V)
        self.R_g = dfx.fem.Function(V)
        self.R_g.interpolate(R_g_expr)
        self.R_g = self.R_g.x.array
        self.R_g = np.reciprocal(self.R_g)
        
    def __str__(self):
        return f'Passive'
        
    def _eval(self, v):              
        return self.R_g * v


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
    
    initial_time_step = True 
    

    def __str__(self):
        return f'Hodgkin–Huxley'
    
    def _eval(self, v):   
        
        # update gating variables
        if self.initial_time_step:            
                                
            # create gating varaibles vectors 
            self.n = 0 * v + self.n_init
            self.m = 0 * v + self.m_init
            self.h = 0 * v + self.h_init            
            
            self.initial_time_step = False            

        else:            
            self.update_gating_variables(v)              

        # conductivities
        g_Na = self.g_Na_leak + self.g_Na_bar*self.m**3*self.h
        g_K  = self.g_K_leak  + self.g_K_bar *self.n**4              
        g_Cl = self.g_Cl_leak

        # ionic currents
        I_ch_Na = g_Na * (v - self.E_Na)
        I_ch_K  = g_K  * (v - self.E_K)
        I_ch_Cl = g_Cl * (v - self.E_Cl)     
                
        return I_ch_Na + I_ch_K + I_ch_Cl      


    def update_gating_variables(self, v):           

        dt_ode = float(self.params['dt'])/self.time_steps_ODE 

        V_M = 1000*(v - self.V_rest) # convert v to mV    
        
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
    
    # Aliev-Panfilov parameters
    mu1 = 0.2  # OC
    mu2 = 0.3  # OC
    k   = 8.0  # OC
    a   = 0.15 # OC
    epsilon = 0.002 # OC
    w_init  = 0.0   # inital state 

    # quantities in Volt for conversion v[V] = 0.1*v - 0.08 or v[mV] = 100*v - 80
    V_min = -80.0
    V_max =  20.0
    conversion_factor = 1.0/(V_max - V_min)       

    # for one time step with u0 = 0 and at t = dt, t[s] = 0.0129t [t.u.] or t[ms] = 12.9t [t.u.]
    # u = dt * I_ion [tu] = 0.0129 * dt * (C * I_ion) [s] -> C = 1/0.0129

    # time_conversion = 1.0/0.0129
    time_factor     = 12.9
    time_conversion = 1.0/time_factor
        
    initial_time_step = True    

    def __str__(self):
        return "Aliev-Panfilov Model"

    def _eval(self, v):

        # conversion from V to adimensional
        v = self.conversion_factor * (v - self.V_min) 

        # update gating variable w
        if self.initial_time_step:            

            # init quantities
            self.w = 0 * v + self.w_init                           
            self.dt_ode = self.time_conversion * float(self.params['dt'])
            
            self.initial_time_step = False    
            
        else:
            self.update_gating_variables(v)        
        
        I_ion = self.k*v*(v - self.a)*(v - 1) + v*self.w 
        
        # convert from tu to ms and mV
        return (self.V_max - self.V_min) * self.time_factor * I_ion
        #return self.time_factor * ((self.V_max - self.V_min) * I_ion  + self.V_min)

    def update_gating_variables(self, v):          
                      
        diff_w = (self.epsilon + self.mu1 * self.w/(v + self.mu2)) * (- self.w - self.k*v*(v - self.a - 1.0))        
        self.w += diff_w * self.dt_ode

# Aliev-Panfilov
class AP_model_2(Ionic_model):
    
    # Aliev-Panfilov parameters
    mu1 = 0.07
    mu2 = 0.3
    k   = 8.0
    a   = 0.1
    epsilon = 0.01
    w_init  = 0.0   # inital state 

    # quantities in Volt for conversion v[V] = 0.1*v - 0.08 or v[mV] = 100*v - 80
    V_min = -85.0
    V_max =  35.0
        
    initial_time_step = True    

    def __str__(self):
        return "Aliev-Panfilov Model 2"

    def _eval(self, v):
        
        # init quantities
        vv = (v - self.V_min) / (self.V_max - self.V_min)

        if self.initial_time_step:            
            self.w = 0 * vv + self.w_init                           
            self.dt_ode = float(self.params['dt'])
            
            self.initial_time_step = False    
        
        I_ion = -(self.V_max - self.V_min)*self.k*(vv*(vv - self.a)*(vv - 1) + vv*self.w)

        if not self.initial_time_step: 
            #v = np.maximum(v,-90.0) # elementwise max min
            #v = np.minimum(v,50.0)
            vv = (v - self.V_min) / (self.V_max - self.V_min)
            self.update_gating_variables(vv)

        return I_ion

    def update_gating_variables(self, v):          
        diff_w = 0.25*(self.epsilon + self.mu1 * self.w/(v + self.mu2)) * (- self.w - self.k*v*(v - self.a - 1.0))        
        self.w += diff_w * self.dt_ode

class Courtemanche_model(Ionic_model):
    """
    Courtemanche et al. (1998) human atrial ionic model
    Translated from the OpenCARP C++ implementation.
    """

    # --- Physical constants ---
    R = 8.3143
    T = 310.0
    F = 96.4867
    gamma = 0.35

    # --- Cell volumes ---
    Volcell = 20100.0
    Voli    = 13668.0
    Volrel  = 96.48
    Volup   = 1109.52

    # --- SR dynamics ---
    tau_tr = 180.0
    tau_u  = 8.0
    tau_f_Ca = 2.0
    k_rel  = 30.0
    k_sat  = 0.1
    dCa_rel = 8.0
    dCaup   = 0.0869565217391304

    # --- Buffers ---
    KmCmdn = 0.00238
    KmCsqn = 0.8
    KmTrpn = 0.0005
    maxCmdn = 0.05
    maxCsqn = 10.0
    maxTrpn = 0.07

    # --- Ion concentrations (initial) ---
    Nai = 11.2
    Ki_init = 139.0
    Cai_init = 1.02e-1 #1.02e-4 
    Ca_up_init = 1.49
    Ca_rel_init = 1.49

    # --- Reversal / Km constants ---
    KmNa = 87.5
    KmNai = 10.0
    KmNa3 = 669921.875
    KmKo  = 1.5
    KmCa  = 1.38
    K_up  = 0.00092

    # --- Conductances ---
    GNa  = 7.8
    GK1  = 0.09
    GKr  = 0.0294
    GKs  = 0.129
    Gto  = 0.1652
    GCaL = 0.1238
    GbNa = 0.000674
    GbCa = 0.00113

    GACh = 0.0
    ACh  = 1e-6

    # --- Pump / exchanger max values ---
    maxINaCa = 1600.0
    maxINaK  = 0.60
    maxIpCa  = 0.275
    maxIup   = 0.005
    maxCaup  = 15.0

    # --- Extra factors (can be varied regionally) ---
    factorGKur = 1.0
    factorGrel = 1.0
    factorGtr  = 1.0
    factorGup  = 1.0
    factorxrGate = 1.0
    factoroaGate = 0.0
    factorhGate  = 0.0
    factormGate  = 0.0

    # --- Extracellular ---
    Ko  = 5.4
    Nao = 140.0
    Cao = 1.8

    # --- Initial gating values ---
    d_init   = 1.37e-4
    f_init   = 0.999
    f_Ca_init= 0.775
    h_init   = 0.965
    j_init   = 0.978
    m_init   = 2.91e-3
    oa_init  = 3.04e-2
    oi_init  = 0.999
    ua_init  = 4.96e-3
    ui_init  = 0.999
    u_init   = 0.0
    v_init   = 1.0
    w_init   = 0.999
    xr_init  = 3.29e-5
    xs_init  = 1.87e-2
    V_init   = -81.2

    # --- Misc constants from C++ code ---
    C_B1c = 0.00705882352941176
    C_B1d = 0.00537112496043221
    C_B1e = 11.5
    C_Fn1 = 9.648e-13
    C_Fn2 = 2.5910306809e-13
    K_Q10 = 3.0

    initial_time_step = True

    def __str__(self):
        return "Courtemanche Model"

    def _compute_currents(self, V):
        """Return dict of all relevant currents at voltage V (no state updates)."""

        V = np.asarray(V, dtype=float)

        E_Na = (self.R * self.T / self.F) * np.log(self.Nao / self.Nai)
        E_K  = (self.R * self.T / self.F) * np.log(self.Ko / self.Ki)

        INa = self.GNa * (V - E_Na) * (self.m**3) * self.h * self.j
        ICaL = self.GCaL * (V - 65.0) * self.d * self.f * self.f_Ca
        IK1 = self.GK1 * (V - E_K) / (1.0 + np.exp(0.07 * (V + 80.0)))
        IKr = self.GKr * (V - E_K) / (1.0 + np.exp((V + 15.0) / 22.4)) * self.xr
        IKs = self.GKs * (V - E_K) * (self.xs**2)
        Ito = self.Gto * (V - E_K) * (self.oa**3) * self.oi
        IKur = (self.factorGKur * (0.005 + 0.05 / (1.0 + np.exp((15.0 - V) / 13.0)))) * \
               (V - E_K) * (self.ua**3) * self.ui
        IbNa = self.GbNa * (V - E_Na)
        #IbCa = self.GbCa * V

        # Pumps/exchangers
        sigma = (np.exp(self.Nao / 67.3) - 1.0) / 7.0
        f_NaK = 1.0 / (1.0 + 0.1245 * np.exp(-0.1 * self.F * V / (self.R * self.T)) +
                       0.0365 * sigma * np.exp(-self.F * V / (self.R * self.T)))
        INaK = (self.maxINaK * f_NaK / (1.0 + (self.KmNai / self.Nai)**1.5)) * \
               self.Ko / (self.Ko + self.KmKo)

        vrow31 = (self.maxINaCa * np.exp(self.gamma * self.F * V / (self.R * self.T)) *
                  (self.Nai**3) * self.Cao) / \
                 ((self.KmNa3 + self.Nao**3) * (self.KmCa + self.Cao) *
                  (1 + self.k_sat * np.exp((self.gamma - 1) * self.F * V / (self.R * self.T))))
        vrow32 = (self.maxINaCa * np.exp((self.gamma - 1) * self.F * V / (self.R * self.T)) *
                  (self.Nao**3)) / \
                 ((self.KmNa3 + self.Nao**3) * (self.KmCa + self.Cao) *
                  (1 + self.k_sat * np.exp((self.gamma - 1) * self.F * V / (self.R * self.T)))) / 1000.0
        INaCa = vrow31 - self.Cai * vrow32

        conCa = self.Cai / 1000.0
        IpCa = (self.maxIpCa*conCa)/(0.0005+conCa) \
                - ((self.GbCa*self.R*self.T)/(2.0*self.F))*np.log(self.Cao/conCa) \
                + self.GbCa*V

        IKACh = (self.GACh * (10.0 / (1.0 + 9.13652 / (self.ACh**0.477811)))) * \
                (0.0517 + 0.4516 / (1.0 + np.exp((V + 59.53) / 17.18))) * (V - E_K)

        IK = IK1 + IKr + IKs + Ito + IKur + IKACh

        Iion = INa + ICaL + IK1 + IKr + IKs + Ito + IKur + IbNa + INaK + INaCa + IpCa + IKACh

        return {
            "Iion": Iion,
            "ICaL": ICaL,
            "INaCa": INaCa,
            "IpCa": IpCa,
            "IK": IK,
            "INaK": INaK
        }

    def _eval(self, V):
        """
        Compute total ionic current (µA/µF) given membrane voltage V (mV).
        Works with scalars or numpy arrays.
        """
        V = np.asarray(V, dtype=float)

        if self.initial_time_step:
            # --- Initialise states ---
            self.dt_ode = float(self.params['dt'])
            self.Ca_rel = self.Ca_rel_init
            self.Ca_up  = self.Ca_up_init
            self.Cai    = self.Cai_init
            self.Ki     = self.Ki_init
            self.d      = self.d_init
            self.f      = self.f_init
            self.f_Ca   = self.f_Ca_init
            self.h      = self.h_init
            self.j      = self.j_init
            self.m      = self.m_init
            self.oa     = self.oa_init
            self.oi     = self.oi_init
            self.ua     = self.ua_init
            self.ui     = self.ui_init
            self.u      = self.u_init
            self.v      = self.v_init
            self.w      = self.w_init
            self.xr     = self.xr_init
            self.xs     = self.xs_init
            self.initial_time_step = False
        else:
            self.update_gating_variables(V)

        currents = self._compute_currents(V)
        return currents["Iion"]

    def update_gating_variables(self, V):
        """
        Update all state variables: Ca_rel, Ca_up, Cai, Ki, gates.
        Euler for concentrations, Rush–Larsen for gates.
        """
        V = np.asarray(V, dtype=float)
        dt = self.dt_ode

        currents = self._compute_currents(V)   # already vectorised

        Iion  = currents["Iion"]
        ICaL  = currents["ICaL"]
        INaCa = currents["INaCa"]
        IpCa  = currents["IpCa"]
        IK    = currents["IK"]
        INaK  = currents["INaK"]

        # --- SR fluxes ---
        Itr   = self.factorGtr * (self.Ca_up - self.Ca_rel) / self.tau_tr
        Irel  = (self.factorGrel * (self.u**2) * self.v * self.k_rel * self.w) * (self.Ca_rel - self.Cai/1000.0)
        dIups = (self.factorGup * self.maxIup) / (1.0 + self.K_up / (self.Cai/1000.0)) \
                - (self.maxIup/self.maxCaup) * self.Ca_up

        # --- Diff equations (Euler) ---
        denom_rel = 1.0 + (self.dCa_rel / (self.Ca_rel + self.KmCsqn)) / (self.Ca_rel + self.KmCsqn)
        diff_Ca_rel = (Itr - Irel) / denom_rel
        diff_Ca_up  = dIups - Itr * self.dCaup

        denom_cai = ((self.maxTrpn*self.KmTrpn)/((self.Cai/1000.0+self.KmTrpn)**2) +
                     (self.maxCmdn*self.KmCmdn)/((self.Cai/1000.0+self.KmCmdn)**2) + 1.0) / self.C_B1c / 1000.0
        diff_Cai   = ((self.C_B1d * ((2*INaCa - IpCa) - ICaL)) - self.C_B1e*dIups + Irel) / denom_cai

        diff_Ki = -(IK - 2.0*INaK) / (self.F * self.Voli)

        # --- Update state variables ---
        self.Ca_rel += diff_Ca_rel * dt
        self.Ca_up  += diff_Ca_up * dt
        self.Cai    += diff_Cai * dt
        self.Ki     += diff_Ki * dt

        # --- Gating updates (Rush–Larsen) ---

        # m-gate (fast sodium activation)
        a_m = np.where(
            np.isclose(V, -47.13),
            3.2,
            0.32 * (V + 47.13) / (1.0 - np.exp(-0.1 * (V + 47.13)))
        )
        b_m = 0.08 * np.exp(-(V - self.factormGate)/11.0)
        tau_m = 1.0 / (a_m + b_m)
        m_inf = a_m / (a_m + b_m)
        self.m = m_inf + (self.m - m_inf) * np.exp(-dt/tau_m)

        # h-gate (fast sodium inactivation)

        a_h = np.where(
        V >= -40.0,
        0.0,
        0.135 * np.exp((V + 80.0 - self.factorhGate) / -6.8)
        )

        b_h = np.where(
            V >= -40.0,
            1.0/0.13 / (1.0 + np.exp(-(V+10.66)/11.1)),
            3.56*np.exp(0.079*V) + 3.1e5*np.exp(0.35*V)
        )

        tau_h = 1.0/(a_h+b_h)
        h_inf = a_h/(a_h+b_h)
        self.h = h_inf + (self.h - h_inf) * np.exp(-dt/tau_h)

        # j-gate (fast sodium slow inactivation)
        a_j = np.where(
            V >= -40.0,
            0.0,
            (((-127140.0*np.exp(0.2444*V)) - (3.474e-5*np.exp(-0.04391*V))) * (V+37.78)) /
            (1.0 + np.exp(0.311*(V+79.23)))
        )

        b_j = np.where(
            V >= -40.0,
            0.3*np.exp(-2.535e-7*V) / (1.0 + np.exp(-0.1*(V+32.0))),
            0.1212*np.exp(-0.01052*V) / (1.0 + np.exp(-0.1378*(V+40.14)))
        )

        tau_j = 1.0 / (a_j + b_j)
        j_inf = a_j / (a_j + b_j)
        self.j = j_inf + (self.j - j_inf) * np.exp(-dt/tau_j)

        # d-gate (L-type Ca activation)
        d_inf = 1.0 / (1.0 + np.exp(-(V+10.0)/8.0))
        tau_d = np.where(
            V == -10.0,
            ((1.0/6.24)/0.035)/2.0,
            ((1.0 - np.exp(-(V+10.0)/6.24)) / (0.035*(V+10.0))) /
            (1.0 + np.exp(-(V+10.0)/6.24))
        )
        self.d = d_inf + (self.d - d_inf) * np.exp(-dt/tau_d)


        # f-gate (L-type Ca inactivation)
        f_inf = 1.0/(1.0+np.exp((V+28.0)/6.9))
        tau_f = 9.0/(0.0197*np.exp(-0.0337*(V+10.0)**2) + 0.02)
        self.f = f_inf + (self.f - f_inf) * np.exp(-dt/tau_f)

        # f_Ca gate
        f_Ca_inf = 1.0/(1.0 + (self.Cai/1000.0)/0.00035)
        self.f_Ca = f_Ca_inf + (self.f_Ca - f_Ca_inf) * np.exp(-dt/self.tau_f_Ca)

        # oa/oi gates (Ito activation/inactivation)
        oa_inf = 1.0/(1.0 + np.exp(-((V+20.47)-self.factoroaGate)/17.54))
        tau_oa = 1.0/(0.65/(np.exp((V+10.0)/-8.5) + np.exp((30.0-V)/59.0)) +
                      0.65/(2.5+np.exp((V+82.0)/17.0))) / self.K_Q10
        self.oa = oa_inf + (self.oa - oa_inf) * np.exp(-dt/tau_oa)

        oi_inf = 1.0/(1.0 + np.exp((V+43.1)/5.3))
        tau_oi = 1.0/(1.0/(18.53+np.exp((V+113.7)/10.95)) +
                      1.0/(35.56+np.exp(-(V+1.26)/7.44))) / self.K_Q10
        self.oi = oi_inf + (self.oi - oi_inf) * np.exp(-dt/tau_oi)

        # ua/ui gates (IKur)
        ua_inf = 1.0/(1.0 + np.exp((V+30.3)/-9.6))
        tau_ua = 1.0/(0.65/(np.exp((V+10.0)/-8.5) + np.exp((V-30.0)/-59.0)) +
                      0.65/(2.5+np.exp((V+82.0)/17.0))) / self.K_Q10
        self.ua = ua_inf + (self.ua - ua_inf) * np.exp(-dt/tau_ua)

        ui_inf = 1.0/(1.0 + np.exp((V-99.45)/27.48))
        tau_ui = 1.0/(1.0/(21.0+np.exp((V-185.0)/-28.0)) +
                      np.exp((V-158.0)/16.0)) / self.K_Q10
        self.ui = ui_inf + (self.ui - ui_inf) * np.exp(-dt/tau_ui)

        # u/v/w gates (Ca release from SR)
        fn = self.C_Fn1*Irel - self.C_Fn2*(self.GCaL*(V-65.0)*self.d*self.f*self.f_Ca - 0.4*INaCa)
        u_inf = 1.0/(1.0 + np.exp((3.4175e-13 - fn)/13.67e-16))
        tau_u = self.tau_u
        self.u = u_inf + (self.u - u_inf) * np.exp(-dt/tau_u)

        v_inf = 1.0 - 1.0/(1.0 + np.exp((6.835e-14 - fn)/13.67e-16))
        tau_v = 1.91 + 2.09/(1.0 + np.exp((3.4175e-13 - fn)/13.67e-16))
        self.v = v_inf + (self.v - v_inf) * np.exp(-dt/tau_v)

        w_inf = 1.0 - 1.0/(1.0 + np.exp((40.0-V)/17.0))
        tau_w = (6.0*(1.0-np.exp((7.9-V)/5.0)))/(1.0+0.3*np.exp((7.9-V)/5.0))/(V-7.9)
        self.w = w_inf + (self.w - w_inf) * np.exp(-dt/tau_w)

        # xr/xs gates (delayed rectifier)
        xr_inf = 1.0/(1.0+np.exp((V+14.1)/-6.5))
        tau_xr = 1.0/(self.factorxrGate*(0.0003*(V+14.1)/(1.0-np.exp(-(V+14.1)/5.0))) +
                      (1.0/self.factorxrGate)*7.3898e-5*(V-3.3328)/(np.exp((V-3.3328)/5.1237)-1.0))
        self.xr = xr_inf + (self.xr - xr_inf) * np.exp(-dt/tau_xr)

        xs_inf = 1.0/np.sqrt(1.0+np.exp((V-19.9)/-12.7))
        tau_xs = 0.5/((4e-5*(V-19.9)/(1.0-np.exp((19.9-V)/17.0))) +
                      (3.5e-5*(V-19.9)/(np.exp((19.9-V)/9.0)-1.0)))
        self.xs = xs_inf + (self.xs - xs_inf) * np.exp(-dt/tau_xs)

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
