from scipy.integrate import solve_ivp
import numpy as np
from src.constants import SPECIES,  DEFAULT_RATES, DEFAULT_INITIAL_CONDITIONS
from src.matrix import STOICHIOMETRIC_MATRIX
from src.model import ode_system

def run_simulation():
    t_span = (0, 1000)
    t_eval = np.linspace(0, 1000, 500)
    
    print("Running ODE solver...")
    solution = solve_ivp(
        fun=lambda t, y: ode_system(t, y, DEFAULT_RATES, STOICHIOMETRIC_MATRIX),
        t_span=t_span,
        y0=DEFAULT_INITIAL_CONDITIONS,
        t_eval=t_eval
    )
    print("Simulation finished.")
    import pandas as pd
    results = pd.DataFrame(solution.y.T, columns=SPECIES)
    results["time"] = solution.t
    return results

if __name__ == "__main__":
    run_simulation()
