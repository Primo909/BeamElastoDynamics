import subprocess
import sys


def run_simulations():
    # Default parameters
    default_params = {
        "L": 1.0,
        "W": 0.1,
        "D": 0.1,
        "NL": 4,
        "NW": 2,
        "ND": 2,
        "E": 1000.0,
        "nu": 0.3,
        "rho": 1.0,
        "eta_m": 0.01,
        "eta_k": 0.01,
        "alpha_m": 0.0,
        "alpha_f": 0.0,
        "total_time": 4.0,
        "cutoff_time_factor": 0.2,
        "num_steps": 50,
    }

    # For 'train' dataset
    initial_force_values = [1.0, 1.5, 2.0, 1.0, 1.5, 2.0]
    cutoff_time_factors = [0.1, 0.1, 0.1, 0.2, 0.2, 0.2]

    for idx, initial_force in enumerate(initial_force_values):
        print(f"\n=== Simulation Set {idx + 1} ===")
        for cutoff_time_factor in cutoff_time_factors:
            params = default_params.copy()
            params.update(
                {
                    "initial_force": initial_force,
                    "cutoff_time_factor": cutoff_time_factor,
                    "mode": "train",
                }
            )
            command = build_command(params)
            print(
                f"\nRunning train simulation with initial_force={initial_force}, cutoff_time_factor={cutoff_time_factor}"
            )
            run_command(command)


def build_command(params):
    """Builds the command list for subprocess.run from parameters."""
    command = ["python", "beam_fea_solver.py"]
    for key, value in params.items():
        command.extend([f"--{key}", str(value)])
    return command


def run_command(command):
    """Runs a command and ensures the output is printed."""
    try:
        # Use subprocess.Popen to capture and print the output in real-time
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        # Read and print the output line by line as it becomes available
        for stdout_line in iter(process.stdout.readline, ""):
            print(stdout_line, end="")
        process.stdout.close()
        return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)
    except Exception as e:
        print(f"An error occurred while running the command: {e}", file=sys.stderr)


if __name__ == "__main__":
    run_simulations()
