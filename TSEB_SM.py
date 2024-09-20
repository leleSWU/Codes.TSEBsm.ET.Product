WATER = 0
CONIFER_E = 1
BROADLEAVED_E = 2
CONIFER_D = 3
BROADLEAVED_D = 4
FOREST_MIXED = 5
SHRUB_C = 6
SHRUB_O = 7
SAVANNA_WOODY = 8
SAVANNA = 9
GRASS = 10
WETLAND = 11
CROP = 12
URBAN = 13
CROP_MOSAIC = 14
SNOW = 15
BARREN = 16
epsilon = 0.622
g0 = 0.1
g1 = 5.2
# Soil heat flux formulation constants
G_CONSTANT = 0
G_RATIO = 1
G_TIME_DIFF = 2
G_TIME_DIFF_SIGMOID = 3

import numpy as np
import meteo_utils as met
import resistances as res
import MO_similarity as MO
import net_radiation as rad
import clumping_index as CI
import Cal_Stress as CS
import Calc_T as CT

# ==============================================================================
# List of constants used in TSEB model and sub-routines
# ==============================================================================
# Change threshold in  Monin-Obukhov lengh to stop the iterations
L_thres = 0.00001
# Change threshold in  friction velocity to stop the iterations
u_thres = 0.00001
# mimimun allowed friction velocity
u_friction_min = 0.01
# Maximum number of interations
ITERATIONS = 100
# kB coefficient
kB = 0.0
# Stephan Boltzmann constant (W m-2 K-4)
sb = 5.670373e-8

# Resistance formulation constants
KUSTAS_NORMAN_1999 = 0
CHOUDHURY_MONTEITH_1988 = 1
MCNAUGHTON_VANDERHURK = 2
CHOUDHURY_MONTEITH_ALPHA_1988 = 3
HAGHIGHI_AND_OR_2015 = 4


def _check_default_parameter_size(parameter, input_array):
    parameter = np.asarray(parameter)
    if parameter.size == 1:
        parameter = np.ones(input_array.shape) * parameter
        return np.asarray(parameter)
    elif parameter.shape != input_array.shape:
        parameter = parameter.reshape(input_array.shape)
        return np.asarray(parameter)
        # raise ValueError(
        #     'dimension mismatch between parameter array and input array with shapes %s and %s' %
        #     (parameter.shape, input_array.shape))
    else:
        return np.asarray(parameter)

def TSEB_SM(
        Tr_K,
        vza,
        T_A_K,
        u,
        ea,
        p,
        sm,
        sm_0,
        sm_max,
        Sn_C,
        Sn_S,
        L_dn,
        LAI,
        h_C,
        emis_C,
        emis_S,
        z_0M,
        d_0,
        z_u,
        z_T,
        landcover,
        RH,
        fm,
        ft,
        leaf_width=0.1,
        z0_soil=0.01,
        alpha_PT=1.26,
        x_LAD=1,
        f_c=1.0,
        f_g=1.0,
        w_C=1.0,
        resistance_form=[0, {}],
        calcG_params=[
            [1],
            0.35],
        UseL=False,
        massman_profile=[0, []]):
    '''Priestley-Taylor TSEB

    Calculates the Priestley Taylor TSEB fluxes using a single observation of
    composite radiometric temperature and using resistances in series.

    Parameters
    ----------
    Tr_K : float
        Radiometric composite temperature (Kelvin).
    vza : float
        View Zenith Angle (degrees).
    T_A_K : float
        Air temperature (Kelvin).
    u : float
        Wind speed above the canopy (m s-1).
    ea : float
        Water vapour pressure above the canopy (mb).
    p : float
        Atmospheric pressure (mb), use 1013 mb by default.
    Sn_C : float
        Canopy net shortwave radiation (W m-2).
    Sn_S : float
        Soil net shortwave radiation (W m-2).
    L_dn : float
        Downwelling longwave radiation (W m-2).
    LAI : float
        Effective Leaf Area Index (m2 m-2).
    h_C : float
        Canopy height (m).
    emis_C : float
        Leaf emissivity.
    emis_S : flaot
        Soil emissivity.
    z_0M : float
        Aerodynamic surface roughness length for momentum transfer (m).
    d_0 : float
        Zero-plane displacement height (m).
    z_u : float
        Height of measurement of windspeed (m).
    z_T : float
        Height of measurement of air temperature (m).
    leaf_width : float, optional
        average/effective leaf width (m).
    z0_soil : float, optional
        bare soil aerodynamic roughness length (m).
    alpha_PT : float, optional
        Priestley Taylor coeffient for canopy potential transpiration,
        use 1.26 by default.
    x_LAD : float, optional
        Campbell 1990 leaf inclination distribution function chi parameter.
    f_c : float, optional
        Fractional cover.
    f_g : float, optional
        Fraction of vegetation that is green.
    w_C : float, optional
        Canopy width to height ratio.
    resistance_form : int, optional
        Flag to determine which Resistances R_x, R_S model to use.

            * 0 [Default] Norman et al 1995 and Kustas et al 1999.
            * 1 : Choudhury and Monteith 1988.
            * 2 : McNaughton and Van der Hurk 1995.

    calcG_params : list[list,float or array], optional
        Method to calculate soil heat flux,parameters.

            * [[1],G_ratio]: default, estimate G as a ratio of Rn_S, default Gratio=0.35.
            * [[0],G_constant] : Use a constant G, usually use 0 to ignore the computation of G.
            * [[2,Amplitude,phase_shift,shape],time] : estimate G from Santanello and Friedl with G_param list of parameters (see :func:`~TSEB.calc_G_time_diff`).
    UseL : float or None, optional
        If included, its value will be used to force the Moning-Obukhov stability length.

    Returns
    -------
    flag : int
        Quality flag, see Appendix for description.
    T_S : float
        Soil temperature  (Kelvin).
    T_C : float
        Canopy temperature  (Kelvin).
    T_AC : float
        Air temperature at the canopy interface (Kelvin).
    L_nS : float
        Soil net longwave radiation (W m-2)
    L_nC : float
        Canopy net longwave radiation (W m-2)
    LE_C : float
        Canopy latent heat flux (W m-2).
    H_C : float
        Canopy sensible heat flux (W m-2).
    LE_S : float
        Soil latent heat flux (W m-2).
    H_S : float
        Soil sensible heat flux (W m-2).
    G : float
        Soil heat flux (W m-2).
    R_S : float
        Soil aerodynamic resistance to heat transport (s m-1).
    R_x : float
        Bulk canopy aerodynamic resistance to heat transport (s m-1).
    R_A : float
        Aerodynamic resistance to heat transport (s m-1).
    u_friction : float
        Friction velocity (m s-1).
    L : float
        Monin-Obuhkov length (m).
    n_iterations : int
        number of iterations until convergence of L.

    # Convert input float scalars to arrays and parameters size
    Tr_K = np.asarray(Tr_K)
    (vza,
     T_A_K,
     sm,
     sm_0,
     sm_max,
     u,
     ea,
     p,
     Sn_C,
     Sn_S,
     L_dn,
     LAI,
     h_C,
     emis_C,
     emis_S,
     z_0M,
     d_0,
     z_u,
     z_T,
     landcover,
     RH,
     fm,
     ft,
     leaf_width,
     z0_soil,
     alpha_PT,
     x_LAD,
     f_c,
     f_g,
     w_C,
     calcG_array) = map(_check_default_parameter_size,
                        [vza,
                         T_A_K,
                         sm,
                         sm_0,
                         sm_max,
                         u,
                         ea,
                         p,
                         Sn_C,
                         Sn_S,
                         L_dn,
                         LAI,
                         h_C,
                         emis_C,
                         emis_S,
                         z_0M,
                         d_0,
                         z_u,
                         z_T,
                         landcover,
                         RH,
                         fm,
                         ft,
                         leaf_width,
                         z0_soil,
                         alpha_PT,
                         x_LAD,
                         f_c,
                         f_g,
                         w_C,
                         calcG_params[1]],
                        [Tr_K] * 31)
    res_params = resistance_form[1]
    resistance_form = resistance_form[0]
    # Create the output variables
    [flag, soil_stress, veg_stress, T_S, T_C, T_AC, Ln_S, Ln_C, LE_C, H_C, LE_S, H_S, G, R_S, R_x, Rn_S, Ln_S,
     R_A, iterations] = [np.zeros(Tr_K.shape) + np.NaN for i in range(19)]

    # iteration of the Monin-Obukhov length
    if isinstance(UseL, bool):
        # Initially assume stable atmospheric conditions and set variables for
        L = np.asarray(np.zeros(T_S.shape) + np.inf)
        max_iterations = ITERATIONS
    else:  # We force Monin-Obukhov lenght to the provided array/value
        L = np.asarray(np.ones(T_S.shape) * UseL)
        max_iterations = 1  # No iteration
    # Calculate the general parameters
    rho = met.calc_rho(p, ea, T_A_K)  # Air density
    c_p = met.calc_c_p(p, ea)  # Heat capacity of air
    z_0H = res.calc_z_0H(z_0M, kB=kB)  # Roughness length for heat transport

    # Calculate LAI dependent parameters for dataset where LAI > 0
    omega0 = CI.calc_omega0_Kustas(LAI, f_c, x_LAD=x_LAD, isLAIeff=True)
    F = np.asarray(LAI / f_c)  # Real LAI
    # Fraction of vegetation observed by the sensor
    f_theta = calc_F_theta_campbell(vza, F, w_C=w_C, Omega0=omega0, x_LAD=x_LAD)

    # Initially assume stable atmospheric conditions and set variables for
    # iteration of the Monin-Obukhov length
    u_friction = MO.calc_u_star(u, z_u, L, d_0, z_0M)
    u_friction = np.asarray(np.maximum(u_friction_min, u_friction))
    L_old = np.ones(Tr_K.shape)
    L_diff = np.asarray(np.ones(Tr_K.shape) * float('inf'))

    # First assume that canopy temperature equals the minumum of Air or
    # radiometric T
    T_C = np.asarray(np.minimum(Tr_K, T_A_K))
    flag, T_S = calc_T_S_SM(Tr_K, T_A_K, f_theta, T_C)
    # Outer loop for estimating stability.
    # Stops when difference in consecutives L is below a given threshold
    g0 = np.ones(Tr_K.shape) * 0.1
    g1 = np.ones(Tr_K.shape) * 5.2
    F1 = np.asarray(LAI / omega0)
    fVeg = 1.0 - np.exp(-0.5 * LAI)
    R_n_sw = Sn_C + Sn_S
    for n_iterations in range(max_iterations):
        i = flag != 255
        # i = np.argwhere(flag != 255)
        if np.all(L_diff[i] < L_thres):
            if L_diff[i].size == 0:
                print("Finished iterations with no valid solution")
            else:
                print("Finished interations with a max. L diff: " + str(np.max(L_diff[i])))
            break
        print("Iteration " + str(n_iterations) +
              ", max. L diff: " + str(np.max(L_diff[i])))
        iterations[np.logical_and(L_diff >= L_thres, flag != 255)] = n_iterations

        # Inner loop to iterativelly reduce alpha_PT in case latent heat flux
        # from the soil is negative. The initial assumption is of potential
        # canopy transpiration.
        flag[np.logical_and(L_diff >= L_thres, flag != 255)] = 0
        LE_S[np.logical_and(L_diff >= L_thres, flag != 255)] = -1
        alpha_PT_rec = np.asarray(alpha_PT + 0.1)
        while np.any(LE_S[i] < 0):
            i = np.logical_and.reduce(
                (LE_S < 0, L_diff >= L_thres, flag != 255))
            # There cannot be negative transpiration from the vegetation
            # Calculate aerodynamic resistances
            R_A_params = {"z_T": z_T[i], "u_friction": u_friction[i], "L": L[i],
                          "d_0": d_0[i], "z_0H": z_0H[i]}
            params = {k: res_params[k][i] for k in res_params.keys()}
            R_x_params = {"u_friction": u_friction[i], "h_C": h_C[i], "d_0": d_0[i],
                          "z_0M": z_0M[i], "L": L[i], "F": F[i], "LAI": LAI[i],
                          "leaf_width": leaf_width[i], "massman_profile": massman_profile,
                          "res_params": params}
            R_S_params = {"u_friction": u_friction[i], "h_C": h_C[i], "d_0": d_0[i],
                          "z_0M": z_0M[i], "L": L[i], "F": F[i], "omega0": omega0[i],
                          "LAI": LAI[i], "leaf_width": leaf_width[i],
                          "z0_soil": z0_soil[i], "z_u": z_u[i],
                          "deltaT": T_S[i] - T_C[i], "massman_profile": massman_profile,
                          'u': u[i], 'rho': rho[i], 'c_p': c_p[i], 'f_cover': f_c[i], 'w_C': w_C[i],
                          "res_params": params}
            res_types = {"R_A": R_A_params, "R_x": R_x_params, "R_S": R_S_params}
            R_A[i], R_x[i], R_S[i] = calc_resistances(resistance_form, res_types)

            # Calculate net longwave radiation with current values of T_C and T_S
            Ln_C[i], Ln_S[i] = rad.calc_L_n_Kustas(
                T_C[i], T_S[i], L_dn[i], LAI[i], emis_C[i], emis_S[i])
            delta_Rn = Sn_C + Ln_C
            Rn_S = Sn_S + Ln_S

            soil_stress[i] = CS.Cal_soil_stress(sm[i], sm_0[i], sm_max[i])

            veg_stress[i] = CS.Cal_veg_stress(g0[i], F1[i], g1[i], soil_stress[i], T_A_K[i], RH[i], R_n_sw[i])
            H_C[i] = calc_H_C_SM(delta_Rn[i], f_g[i], T_A_K[i], p[i], c_p[i], alpha_PT_rec[i], veg_stress[i])
            T_C[i] = CT.CalcT_C_Series(Tr_K[i], T_A_K[i], R_A[i], R_x[i], R_S[i], f_theta[i], H_C[i], rho[i], c_p[i])

            G[i] = calc_G([calcG_params[0], calcG_array], Rn_S, i)
          
            H_S[i] = CalcH_S_PT(Rn_S[i], G[i], f_g[i], T_A_K[i], p[i], c_p[i], alpha_PT[i], soil_stress[i])
            
            T_S[i] = CT.CalcT_S_Series(Tr_K[i], T_A_K[i], R_A[i], R_x[i], R_S[i], f_theta[i], H_S[i], rho[i], c_p[i])
            # Recalculate soil resistance using new soil temperature
            params = {k: res_params[k][i] for k in res_params.keys()}
            R_S_params = {"u_friction": u_friction[i], "h_C": h_C[i], "d_0": d_0[i],
                          "z_0M": z_0M[i], "L": L[i], "F": F[i], "omega0": omega0[i],
                          "LAI": LAI[i], "leaf_width": leaf_width[i],
                          "z0_soil": z0_soil[i], "z_u": z_u[i],
                          "deltaT": T_S[i] - T_C[i], "massman_profile": massman_profile,
                          'u': u[i], 'rho': rho[i], 'c_p': c_p[i], 'f_cover': f_c[i], 'w_C': w_C[i],
                          "res_params": params}
            _, _, R_S[i] = calc_resistances(resistance_form, {"R_S": R_S_params})
        
            i = np.logical_and.reduce(
                (LE_S < 0, L_diff >= L_thres, flag != 255))

            # Get air temperature at canopy interface
            T_AC[i] = ((T_A_K[i] / R_A[i] + T_S[i] / R_S[i] + T_C[i] / R_x[i])
                       / (1.0 / R_A[i] + 1.0 / R_S[i] + 1.0 / R_x[i]))
            # Calculate soil fluxes
            H_S[i] = rho[i] * c_p[i] * (T_S[i] - T_AC[i]) / R_S[i]

            # Estimate latent heat fluxes as residual of energy balance at the
            # soil and the canopy
            LE_S[i] = Rn_S[i] - G[i] - H_S[i]
            LE_C[i] = delta_Rn[i] - H_C[i]

            # Special case if there is no transpiration from vegetation.
            # In that case, there should also be no evaporation from the soil
            # and the energy at the soil should be conserved.
            # See end of appendix A1 in Guzinski et al. (2015).
            noT = np.logical_and(i, LE_C == 0)
            H_S[noT] = np.minimum(H_S[noT], Rn_S[noT] - G[noT])
            G[noT] = np.maximum(G[noT], Rn_S[noT] - H_S[noT])
            LE_S[noT] = 0

            # Calculate total fluxes
            H = np.asarray(H_C + H_S)
            LE = np.asarray(LE_C + LE_S)
            # Now L can be recalculated and the difference between iterations
            # derived
            if isinstance(UseL, bool):
                L[i] = MO.calc_L(
                    u_friction[i],
                    T_A_K[i],
                    rho[i],
                    c_p[i],
                    H[i],
                    LE[i])
                # Calculate again the friction velocity with the new stability
                # correctios
                u_friction[i] = MO.calc_u_star(
                    u[i], z_u[i], L[i], d_0[i], z_0M[i])
                u_friction = np.asarray(np.maximum(u_friction_min, u_friction))

        if isinstance(UseL, bool):
            L_diff = np.asarray(np.fabs(L - L_old) / np.fabs(L_old))
            L_diff[np.isnan(L_diff)] = float('inf')
            L_old = np.array(L)
            L_old[L_old == 0] = 1e-36

    (flag,
     T_S,
     T_C,
     T_AC,
     L_nS,
     L_nC,
     LE_C,
     H_C,
     LE_S,
     H_S,
     G,
     R_S,
     R_x,
     R_A,
     Rn_S,
     Ln_S,
     u_friction,
     L,
     n_iterations) = map(np.asarray,
                         (flag,
                          T_S,
                          T_C,
                          T_AC,
                          Ln_S,
                          Ln_C,
                          LE_C,
                          H_C,
                          LE_S,
                          H_S,
                          G,
                          R_S,
                          R_x,
                          R_A,
                          Rn_S,
                          Ln_S,
                          u_friction,
                          L,
                          iterations))

    return flag, T_S, T_C, T_AC, L_nS, L_nC, LE_C, H_C, LE_S, H_S, G, R_S, R_x, R_A, Rn_S, Ln_S, u_friction, L, n_iterations
