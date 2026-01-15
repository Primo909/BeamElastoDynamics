import subprocess
import sys


def run_simulations():
    # Default parameters
    default_params = {
        "L": 1.0,
        "W": 0.5,
        "D": 0.5,
        "NL": 16,
        "NW": 8,
        "ND": 8,
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
    initial_force_values = [10]
    cutoff_time_factors = [0.1]

    for initial_force in initial_force_values:
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

    # For 'test_extr_force'
    params = default_params.copy()
    params.update(
        {"initial_force": 1.5, "cutoff_time_factor": 0.5, "mode": "test_extr_force"}
    )
    command = build_command(params)
    print(
        f"\nRunning test_extr_force simulation with initial_force=1.5, cutoff_time_factor=0.5"
    )
    run_command(command)

    # For 'test_extr_geo'
    params = default_params.copy()
    params.update(
        {
            "L": 2.0,
            "W": 0.5,
            "D": 0.5,
            "NL": 64,
            "NW": 8,
            "ND": 8,
            "initial_force": 1.0,
            "cutoff_time_factor": 0.2,
            "mode": "test_extr_geo",
        }
    )
    command = build_command(params)
    print(f"\nRunning test_extr_geo simulation with L=1.5, W=0.2, D=0.2")
    run_command(command)

    # For 'test_extr_desc'
    params = default_params.copy()
    params.update(
        {
            "NL": 16,
            "NW": 4,
            "ND": 4,
            "initial_force": 1.0,
            "cutoff_time_factor": 0.2,
            "mode": "test_extr_desc",
        }
    )
    command = build_command(params)
    print(f"\nRunning test_extr_desc simulation with NL=16, NW=4, ND=4")
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
