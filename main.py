import argparse
import os
import json
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from src.constants import SPECIES
from src.matrix import STOICHIOMETRIC_MATRIX
from src.model import ode_system
from src.plotting import plot_results

def run_simulation(params, initial_conditions, t_span, t_eval):
    """Runs a single ODE simulation."""
    solution = solve_ivp(
        fun=lambda t, y: ode_system(t, y, params, STOICHIOMETRIC_MATRIX),
        t_span=t_span,
        y0=initial_conditions,
        t_eval=t_eval
    )
    results = pd.DataFrame(solution.y.T, columns=SPECIES)
    results["time"] = solution.t
    return results

def main():
    """Main function to run the commodity price prediction workflow."""
    parser = argparse.ArgumentParser(description="Deactivation Kinetics Simulation")
    parser.add_argument('--config_path', type=str, default="config/default_params.json", help="Path to the JSON config file.")
    parser.add_argument('--output_dir', type=str, default="results", help="Directory to save simulation results.")
    
    # Arguments for parameter sweep
    parser.add_argument('--sweep_param', type=str, default=None, help="Name of the rate constant to sweep (e.g., 'k1').")
    parser.add_argument('--sweep_start', type=float, default=0.1, help="Start value for the parameter sweep.")
    parser.add_argument('--sweep_end', type=float, default=2.0, help="End value for the parameter sweep.")
    parser.add_argument('--sweep_steps', type=int, default=5, help="Number of steps in the parameter sweep.")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config_path, "r") as f:
        base_params = json.load(f)

    if args.sweep_param:
        print(f"--- Running Parameter Sweep for '{args.sweep_param}' ---")
        sweep_values = np.linspace(args.sweep_start, args.sweep_end, args.sweep_steps)
        
        for i, value in enumerate(sweep_values):
            print(f"Running sweep {i+1}/{args.sweep_steps}: {args.sweep_param} = {value:.4f}")
            
            # Create a copy of params and update the sweep parameter
            run_params = base_params['rate_constants'].copy()
            if args.sweep_param not in run_params:
                print(f"Error: sweep parameter '{args.sweep_param}' not in rate constants.")
                return
            run_params[args.sweep_param] = value
            
            # Run simulation
            results = run_simulation(run_params, base_params['initial_conditions'], (0, 1000), np.linspace(0, 1000, 500))
            
            # Save results
            output_csv_path = os.path.join(args.output_dir, f"results_{args.sweep_param}_{value:.4f}.csv")
            output_plot_path = os.path.join(args.output_dir, f"plot_{args.sweep_param}_{value:.4f}.png")
            results.to_csv(output_csv_path, index=False)
            plot_results(results, ["C+", "iC4+", "Cn="], output_plot_path)
            
        print("\nSweep complete.")

    else:
        print("--- Running Single Simulation ---")
        results = run_simulation(base_params['rate_constants'], base_params['initial_conditions'], (0, 1000), np.linspace(0, 1000, 500))
        
        output_csv_path = os.path.join(args.output_dir, "results_single_run.csv")
        output_plot_path = os.path.join(args.output_dir, "plot_single_run.png")
        results.to_csv(output_csv_path, index=False)
        plot_results(results, ["C+", "iC4+", "Cn="], output_plot_path)
        print(f"Single run complete. Results saved in '{args.output_dir}'")

if __name__ == "__main__":
    main()