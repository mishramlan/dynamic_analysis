# constitutive.py

import numpy as np

def dev(sigma):
    """Deviatoric part of 6-vector stress [s11,s22,s33,s12,s23,s13]."""
    s1,s2,s3,s12,s23,s13 = sigma
    mean = (s1+s2+s3)/3
    return np.array([s1-mean, s2-mean, s3-mean, s12, s23, s13])

def von_mises_eq(sigma):
    s = dev(sigma)
    return np.sqrt(1.5 * np.dot(s, s))

def update_viscoplastic(eps_new, eps_p_old, 
                        D, sigma_y, eta, dt):
    """
    Perzyna visco-plastic update.
      sigma_old: 6-vector at time n (unused here)
      eps_new:   total strain 6-vector at time n+1
      eps_p_old: plastic strain 6-vector at time n
      D:         6×6 elasticity matrix
      sigma_y:   yield stress
      eta:       viscosity
      dt:        time step
    Returns (sigma_new, eps_p_new).
    """
    # 1) trial stress
    sigma_trial = D.dot(eps_new - eps_p_old)
    s_trial = dev(sigma_trial)
    sigma_eq = np.linalg.norm(s_trial) * np.sqrt(1.5)
    
    if sigma_eq <= sigma_y:
        return sigma_trial, eps_p_old
    
    # 2) Perzyna flow
    gamma_dot = (sigma_eq - sigma_y) / eta
    dgamma = gamma_dot * dt
    
    # flow direction
    n_dir = (3/2) * s_trial / (np.linalg.norm(s_trial) + 1e-16)
    
    eps_p_new = eps_p_old + dgamma * n_dir
    sigma_new = D.dot(eps_new - eps_p_new)
    return sigma_new, eps_p_new
