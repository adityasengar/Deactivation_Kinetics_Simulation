import numpy as np

def calculate_rate_vector(y, k):
    """Calculates the reaction rate vector 'v'."""
    C3_plus, iC4_plus, H_plus, H_plus_2, C_plus_species, iC4, C, C3, C7, Cn_eq = y
    
    # Rate expressions v from SI, page 2
    v = np.array([
        k['k1'] * C3,          # v1
        k['k2'] * iC4_plus,    # v2
        k['k3'] * iC4_plus,    # v3
        k['k4'] * C_plus_species,      # v4
        k['k5'] * C_plus_species,      # v5
        k['k6'] * H_plus * iC4, # v6
        k['k7'] * H_plus * C,   # v7
        0,                     # v8 is not in S matrix
        k['k9'] * H_plus,      # v9
        k['k10'] * C,          # v10
        k['k11'] * H_plus * Cn_eq # v11
    ])
    return v

"""The system of ODEs: dX/dt = S * v."""(t, y, k, S):
    """The system of ODEs: dX/dt = S * v."""
    v = calculate_rate_vector(y, k)
    return np.dot(S, v)
