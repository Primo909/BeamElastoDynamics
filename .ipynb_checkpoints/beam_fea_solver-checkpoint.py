from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from fenics import project
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import os
import argparse

def extract_edges(mesh):
    # Extract the element-to-node connectivity from the mesh
    mesh_connectivity = mesh.cells()  # Get element-to-node mapping

    # Use a set to store unique edges (each edge as a tuple of sorted node indices)
    edges = set()

    # Iterate through each element (cell)
    for element in mesh_connectivity:
        # For each element, find pairs of nodes (edges) and store them
        for i in range(len(element)):
            for j in range(i + 1, len(element)):
                # Add the edge as a sorted tuple to avoid duplicate (i, j) and (j, i)
                edges.add(tuple(sorted((element[i], element[j]))))

    # Convert the set of edges to a list of edges
    edge_index = list(edges)

    # Convert edge_index to a tensor format for PyTorch Geometric
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    bidir_edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return bidir_edge_index,mesh_connectivity

def extract_node_masses(u_,du, rho):
    # Extract node wise lumped masses
    m_mass = rho * inner(u_, du) * dx
    M = assemble(m_mass)
    # Initialize the vector 'diag' to have the correct size and map
    diag = Vector()
    M.init_vector(diag, 0)  # '0' indicates the column map of the matrix

    # Extract the diagonal of the mass matrix into 'diag'
    M.get_diagonal(diag)

    # Get the local array of diagonal entries (lumped masses per DOF)
    lumped_masses_per_dof = diag.get_local()
    lumped_masses_nodes = lumped_masses_per_dof.reshape(-1,3) 
    return lumped_masses_nodes   

def extract_mesh_bc(bc,V):
    # Get constrained DOF indices from Dirichlet boundary conditions
    bc_dofs_dict = bc.get_boundary_values()
    bc_dof_indices = np.array(list(bc_dofs_dict.keys()), dtype=int)

    # Number of components (e.g., 3 for 3D problems)
    num_components = V.num_sub_spaces() or V.ufl_element().value_size()
    num_dofs = V.dim()
    num_nodes = num_dofs // num_components

    # Map DOF indices to node indices
    node_indices = bc_dof_indices // num_components

    # Initialize the array to indicate if a node is fixed (1) or not (0)
    is_fixed = np.zeros(num_nodes, dtype=int)

    # Mark the nodes as fixed
    is_fixed[node_indices] = 1
    node_scalar = torch.tensor(is_fixed, dtype=torch.float).reshape(-1,1)    
    return node_scalar

def plot_check(time, u_tip, energies, save_path):
    
    if MPI.comm_world.rank == 0:  # Only rank 0 should handle plotting and saving the figure
        # Create a figure with 2 subplots
        fig, ax = plt.subplots(2, 1, figsize=(8, 6))

        # Subplot 1: Tip displacement evolution
        ax[0].plot(time, u_tip)
        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("Tip displacement")
        ax[0].set_title("Tip Displacement Evolution")

        # Subplot 2: Energies evolution
        ax[1].plot(time, energies)
        ax[1].legend(("Elastic", "Kinetic", "Damping", "Total"))
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("Energies")
        ax[1].set_title("Energies Evolution")

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save the figure to the specified path
        plt.savefig(save_path)
        print('plot saved to :',save_path)

        # Show the plot (optional, depending on whether you need this in a headless environment)
        plt.show()

def save_graphs(lst_graph_tstep,save_path):
    gph_dataloader = DataLoader(lst_graph_tstep, batch_size=1, shuffle=False)
    torch.save(gph_dataloader, save_path)
    print(f'saved graphs(dataloader) to {save_path}')



def fea_simulation(L, W, D, NL, NW, ND, elastic_params, density, damping_params, newmark_params,
                   initial_force=1.0, cutoff_time_factor=1/5, total_time=4.0, num_steps=50,mode='train'):
    # Form compiler options
    parameters["form_compiler"]["cpp_optimize"] = True
    parameters["form_compiler"]["optimize"] = True

    # Elastic parameters
    E, nu = elastic_params
    mu = Constant(E / (2.0 * (1.0 + nu)))
    lmbda = Constant(E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))

    # Mass density
    rho = Constant(density)

    # Rayleigh damping coefficients
    eta_m_val, eta_k_val = damping_params
    eta_m = Constant(eta_m_val)
    eta_k = Constant(eta_k_val)

    # Generalized-alpha method parameters
    alpha_m_val, alpha_f_val = newmark_params
    alpha_m = Constant(alpha_m_val)
    alpha_f = Constant(alpha_f_val)
    gamma = Constant(0.5 + alpha_f - alpha_m)
    beta = Constant((gamma + 0.5)**2 / 4.)

    # Define Mesh
    mesh = BoxMesh(Point(0., 0., 0.), Point(L, W, D), NL, NW, ND)

    # Sub domain for clamp at left end
    def left(x, on_boundary):
        return near(x[0], 0.) and on_boundary

    # Sub domain for force application at right end
    def right(x, on_boundary):
        return near(x[0], L) and on_boundary

    # Define function space for displacement, velocity and acceleration
    V = VectorFunctionSpace(mesh, "CG", 1)
    # Define function space for stresses
    Vsig = TensorFunctionSpace(mesh, "DG", 0)

    #------------------------------------------------------
    bidir_edge_index,mesh_connectivity = extract_edges(mesh)
    #------------------------------------------------------

    # Test and trial functions
    du = TrialFunction(V)
    u_ = TestFunction(V)
    # Current (unknown) displacement
    u = Function(V, name="Displacement")
    # Fields from previous time step (displacement, velocity, acceleration)
    u_old = Function(V)
    v_old = Function(V)
    a_old = Function(V)

    # Create mesh function over the cell facets
    boundary_subdomains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary_subdomains.set_all(0)
    force_boundary = AutoSubDomain(right)
    force_boundary.mark(boundary_subdomains, 3)

    # Define measure for boundary condition integral
    dss = ds(subdomain_data=boundary_subdomains)

    # Set up boundary condition at left end
    zero = Constant((0.0, 0.0, 0.0))
    bc = DirichletBC(V, zero, left)

    #------------------------------------------------------
    # Extract node-wise boundary conditions
    node_scalar = extract_mesh_bc(bc,V)
    #------------------------------------------------------

    # Time-stepping parameters
    T = total_time
    Nsteps = num_steps
    dt = Constant(T / Nsteps)

    # Define Force
    p0 = initial_force
    cutoff_Tc = T * cutoff_time_factor
    # Define the loading as an expression depending on t
    p = Expression(("0", "t <= tc ? p0 * t / tc : 0", "0"), t=0, tc=cutoff_Tc, p0=p0, degree=0)

    # Stress tensor
    def sigma(r):
        return 2.0 * mu * sym(grad(r)) + lmbda * tr(sym(grad(r))) * Identity(len(r))

    # Mass form
    def m(u, u_):
        return rho * inner(u, u_) * dx

    # Elastic stiffness form
    def k(u, u_):
        return inner(sigma(u), sym(grad(u_))) * dx

    # Rayleigh damping form
    def c(u, u_):
        return eta_m * m(u, u_) + eta_k * k(u, u_)

    # Work of external forces
    def Wext(u_):
        return dot(u_, p) * dss(3)

    # Update formula for acceleration
    def update_a(u, u_old, v_old, a_old, ufl=True):
        if ufl:
            dt_ = dt
            beta_ = beta
        else:
            dt_ = float(dt)
            beta_ = float(beta)
        return (u - u_old - dt_ * v_old) / beta_ / dt_**2 - (1 - 2 * beta_) / (2 * beta_) * a_old

    # Update formula for velocity
    def update_v(a, u_old, v_old, a_old, ufl=True):
        if ufl:
            dt_ = dt
            gamma_ = gamma
        else:
            dt_ = float(dt)
            gamma_ = float(gamma)
        return v_old + dt_ * ((1 - gamma_) * a_old + gamma_ * a)

    def update_fields(u, u_old, v_old, a_old):
        """Update fields at the end of each time step."""
        # Get vectors (references)
        u_vec, u0_vec = u.vector(), u_old.vector()
        v0_vec, a0_vec = v_old.vector(), a_old.vector()

        # use update functions using vector arguments
        a_vec = update_a(u_vec, u0_vec, v0_vec, a0_vec, ufl=False)
        v_vec = update_v(a_vec, u0_vec, v0_vec, a0_vec, ufl=False)

        # Update (u_old <- u)
        v_old.vector()[:] = v_vec
        a_old.vector()[:] = a_vec
        u_old.vector()[:] = u.vector()

    def avg(x_old, x_new, alpha):
        return alpha * x_old + (1 - alpha) * x_new

    # Residual
    a_new = update_a(du, u_old, v_old, a_old, ufl=True)
    v_new = update_v(a_new, u_old, v_old, a_old, ufl=True)
    res = m(avg(a_old, a_new, alpha_m), u_) + c(avg(v_old, v_new, alpha_f), u_) \
        + k(avg(u_old, du, alpha_f), u_) - Wext(u_)
    a_form = lhs(res)
    L_form = rhs(res)

    # Define solver for reusing factorization
    K, res_assembled = assemble_system(a_form, L_form, bc)
    solver = LUSolver(K, "mumps")
    solver.parameters["symmetric"] = True

    # Time-stepping
    time = np.linspace(0, T, Nsteps + 1)
    u_tip = np.zeros((Nsteps + 1,))
    energies = np.zeros((Nsteps + 1, 4))
    E_damp = 0
    E_ext = 0
    sig = Function(Vsig, name="sigma")

    #------------------------------------------------------
    lumped_masses_nodes = extract_node_masses(u_,du, rho)
    #------------------------------------------------------

    def local_project(v, V, u=None):
        """Element-wise projection using LocalSolver"""
        dv = TrialFunction(V)
        v_ = TestFunction(V)
        a_proj = inner(dv, v_) * dx
        b_proj = inner(v, v_) * dx
        solver = LocalSolver(a_proj, b_proj)
        solver.factorize()
        if u is None:
            u = Function(V)
            solver.solve_local_rhs(u)
            return u
        else:
            solver.solve_local_rhs(u)
            return

    lst_graph_tstep = []

    # Create a VectorFunctionSpace over the mesh
    V_p = VectorFunctionSpace(mesh, 'CG', 1)
    V_stress_nodes = TensorFunctionSpace(mesh, "CG", 1)

    # Create a Function in the vector function space
    p_function = Function(V_p)  

    initial_coordinates = mesh.coordinates()  

    for i in range(Nsteps):
        dt_i = time[i+1] - time[i]
        t = time[i+1]
        print("Time: ", t)

        # Forces are evaluated at t_{n+1-alpha_f}=t_{n+1}-alpha_f*dt
        p.t = t - float(alpha_f) * dt_i

        # Solve for new displacement
        res_assembled = assemble(L_form)
        bc.apply(res_assembled)
        solver.solve(K, u.vector(), res_assembled)

        # Update old fields with new quantities
        update_fields(u, u_old, v_old, a_old)

        # Compute stresses and save to file
        local_project(sigma(u), Vsig, sig)

        # projected node wise stress
        stress_nodewise = project(sig, V_stress_nodes)

        # Apply the force p as a Dirichlet boundary condition on the boundary with marker 3
        bcs_force = DirichletBC(V_p, p, boundary_subdomains, 3)  # '3' is the marker for the force boundary

        # Apply the boundary condition to p_function
        bcs_force.apply(p_function.vector())

        # Extract nodal values
        p_nodal_values = p_function.vector().get_local().reshape(-1, 3)

        # Convert to PyTorch tensor
        node_force_t = torch.tensor(p_nodal_values, dtype=torch.float)    

        deformed_coordinates = initial_coordinates + u.vector().get_local().reshape(-1, 3)
        pos_t = torch.tensor(deformed_coordinates, dtype=torch.float)  # position at current time 
        vel_t = torch.tensor(v_old.vector().get_local().reshape(-1, 3), dtype=torch.float)  # Velocity at current time
        acc_t = torch.tensor(a_old.vector().get_local().reshape(-1, 3), dtype=torch.float)  # Acceleration at current time
        sig_t = torch.tensor(stress_nodewise.vector().get_local().reshape(-1, 9), dtype=torch.float)  # Stress (node-wise)   

        # Create graph data object with node positions (x), edge index, velocity, acceleration, stress, and time
        graph = Data(x=pos_t,  # Node positions
                    edge_index=bidir_edge_index,  # Precomputed edge index
                    x_vel_t=vel_t,  # Current velocity
                    y_acc_t=acc_t,  # Current acceleration
                    y_sig_t=sig_t,  # Current stress
                    x_node_force = node_force_t,
                    x_bc = node_scalar,
                    y_node_lumped_masses = lumped_masses_nodes,
                    x_element_connectivity = mesh_connectivity,
                    time=torch.tensor([t], dtype=torch.float))  # Global time value
        lst_graph_tstep.append(graph)
        #print(f'saved : {graph}')
    

        p.t = t

        # Record tip displacement and compute energies
        # Note: Only works in serial
        if MPI.comm_world.size == 1:
            u_tip[i+1] = u(L, W, 0.)[1]  # Tip displacement in y-direction
        E_elas = assemble(0.5 * k(u_old, u_old))
        E_kin = assemble(0.5 * m(v_old, v_old))
        E_damp += dt_i * assemble(c(v_old, v_old))
        E_tot = E_elas + E_kin + E_damp
        energies[i+1, :] = np.array([E_elas, E_kin, E_damp, E_tot])

        # Append data to lst_graph_tstep if needed
        # Currently, lst_graph_tstep is not populated in this code
        # You can add code here if you wish to store graphs per time step
    
    # Construct simulation name with 3 significant digits for float values
    simulation_name = (f'L{L}_W{W}_D{D}_NL{NL}_NW{NW}_ND{ND}_'f'E{E}_nu{nu}_rho{density}_'
                    f'em{eta_m_val}_ek{eta_k_val}_Pi{initial_force}_'
                    f'T{total_time}_Tc{cutoff_time_factor * total_time}_Nsteps{num_steps}')

    # Define the root save path for results
    save_path = './Results'

    # Create the main directory if it does not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Define and create subdirectories for saving graphs and plot_check results
    save_path_graph = os.path.join(save_path,mode, 'graphs')
    save_path_plot_check = os.path.join(save_path,mode, 'plot_check')

    os.makedirs(save_path_graph, exist_ok=True)
    os.makedirs(save_path_plot_check, exist_ok=True)

    # Define file paths for plot and graph data using the simulation_name
    plot_file_path = os.path.join(save_path_plot_check, f'plot_{simulation_name}.png')
    graph_file_path = os.path.join(save_path_graph, f'graphs{simulation_name}.pt')  # Assuming graph data is saved in .pt format

    # Call the plot_check function and save the plot with the simulation name
    plot_check(time, u_tip, energies, plot_file_path)

    # Save the graphs using the simulation name
    save_graphs(lst_graph_tstep, graph_file_path)
    
if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description='FEA Simulation Parameters')

    # Geometrical and discretization parameters
    parser.add_argument('--L', type=float, default=1.0, help='Length of the beam (default: 1.0)')
    parser.add_argument('--W', type=float, default=0.1, help='Width of the beam (default: 0.1)')
    parser.add_argument('--D', type=float, default=0.04, help='Depth of the beam (default: 0.04)')
    parser.add_argument('--NL', type=int, default=8, help='Number of elements along length (default: 8)')
    parser.add_argument('--NW', type=int, default=2, help='Number of elements along width (default: 2)')
    parser.add_argument('--ND', type=int, default=2, help='Number of elements along depth (default: 2)')

    # Material properties
    parser.add_argument('--E', type=float, default=1000.0, help="Young's modulus (default: 1000.0)")
    parser.add_argument('--nu', type=float, default=0.3, help="Poisson's ratio (default: 0.3)")
    parser.add_argument('--rho', type=float, default=1.0, help='Density (default: 1.0)')

    # Damping parameters
    parser.add_argument('--eta_m', type=float, default=0.01, help='Mass proportional damping (default: 0.01)')
    parser.add_argument('--eta_k', type=float, default=0.01, help='Stiffness proportional damping (default: 0.01)')

    # Newmark method parameters
    parser.add_argument('--alpha_m', type=float, default=0.0, help='Alpha mass (default: 0.0)')
    parser.add_argument('--alpha_f', type=float, default=0.0, help='Alpha force (default: 0.0)')

    # Simulation parameters
    parser.add_argument('--initial_force', type=float, default=1.0, help='Initial force (default: 1.0)')
    parser.add_argument('--cutoff_time_factor', type=float, default=1/5, help='Cutoff time factor (default: 0.2)')
    parser.add_argument('--total_time', type=float, default=4.0, help='Total simulation time (default: 4.0)')
    parser.add_argument('--num_steps', type=int, default=50, help='Number of time steps (default: 50)')

    #flag
    parser.add_argument('--mode', type=str, default='train', help='subfolder name e.g. test,train')

    # Parse the arguments
    args = parser.parse_args()

    # Extract parameters from args
    L = args.L
    W = args.W
    D = args.D
    NL = args.NL
    NW = args.NW
    ND = args.ND
    E = args.E
    nu = args.nu
    rho = args.rho
    eta_m = args.eta_m
    eta_k = args.eta_k
    alpha_m = args.alpha_m
    alpha_f = args.alpha_f
    initial_force = args.initial_force
    cutoff_time_factor = args.cutoff_time_factor
    total_time = args.total_time
    num_steps = args.num_steps
    mode = args.mode

    # Pack parameters into tuples
    elastic_params = (E, nu)
    density = rho
    damping_params = (eta_m, eta_k)
    newmark_params = (alpha_m, alpha_f)

    # Run simulation
    fea_simulation(L, W, D, NL, NW, ND, elastic_params, density, damping_params, newmark_params,
                           initial_force, cutoff_time_factor, total_time, num_steps,mode)


