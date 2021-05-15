import jsonfrom scipy.integrate import solve_ivp
import numpy as np
from src.constants import SPECIES,  params["rate_constants"], params["initial_conditions"]
from src.matrix import STOICHIOMETRIC_MATRIX
from src.model import ode_system

def from src.plotting import plot_results
    results = with open("config/default_params.json", "r") as f: params = json.load(f)\n    results = run_simulation(params)
    plot_results(results, ["C+", "iC4+", "Cn="]):
    t_span = (0, 1000)
    t_eval = np.linspace(0, 1000, 500)
    
    print("Running ODE solver...")
    solution = solve_ivp(
        fun=lambda t, y: ode_system(t, y, params["rate_constants"], STOICHIOMETRIC_MATRIX),
        t_span=t_span,
        y0=params["initial_conditions"],
        t_eval=t_eval
    )
    print("Simulation finished.")
    import pandas as pd
    results = pd.DataFrame(solution.y.T, columns=SPECIES)
    results["time"] = solution.t
    results.to_csv("results.csv", index=False)
    print("Results saved to results.csv")
    return results

if __name__ == "__main__":
    from src.plotting import plot_results
    results = with open("config/default_params.json", "r") as f: params = json.load(f)\n    results = run_simulation(params)
    plot_results(results, ["C+", "iC4+", "Cn="])
