import math
import random
import time
import numpy
import os

#%% define the simulation box
x_length = 40  # unit of length is set to Angstrom
y_z_length = 40
bond_length = 1.162
num_molecule = 10
T = 600
beta_mu = -16
mu = beta_mu * T
LAMBDA = 2.6319 / math.sqrt(T)

#%% record file generate
def paths_define(folder=None):
    if folder is None:
        folder = "."
    global txt_save,txt_save_0 ,txt_read,density_save,external_potential_file,log_file,external_potential_params_file
    global full_density_profile_file,dis_density_profile_file_txt,ang_density_profile_file_txt
    txt_save_0 = f'{folder}/configuration_0.txt'
    txt_save = f'{folder}/configuration_now.txt'
    txt_read = f'{folder}/configuration_now.txt'
    density_save = f'{folder}/density_profile.txt'
    external_potential_params_file= f'{folder}/parameters.txt'
    external_potential_file = f'{folder}/external_potential.txt'
    log_file = f'{folder}/log.txt'
    full_density_profile_file=f'{folder}/full_density_profile.npy'
    dis_density_profile_file_txt=f'{folder}/full_x_density_profile.txt'
    ang_density_profile_file_txt=f'{folder}/full_ang_density_profile.txt'

#%% configurations
# define the configuration savement and readment
def save_configuration_to_txt(configuration, filename):
    with open(filename, 'w') as file:
        for O1, C, O2 in configuration:
            file.write(f"{O1[0]} {O1[1]} {O1[2]} {C[0]} {C[1]} {C[2]} {O2[0]} {O2[1]} {O2[2]}\n")

def read_configuration_from_txt(filename):
    configuration = []
    with open(filename, 'r') as file:
        for line in file:
            x_O1, y_O1, z_O1, x_C, y_C, z_C, x_O2, y_O2, z_O2 = map(float, line.split())
            configuration.append(((x_O1, y_O1, z_O1), (x_C, y_C, z_C), (x_O2, y_O2, z_O2)))
    return configuration

# define the initialization of configuration
def generate_molecules(x_length, y_z_length, bond_length, num_molecule):
    configuration = []

    while len(configuration) < num_molecule:
        # random start point O_1
        x_O1 = random.uniform(0, x_length)
        y_O1 = random.uniform(0, y_z_length)
        z_O1 = random.uniform(0, y_z_length)

        # random direction
        theta = random.uniform(0, math.pi)  # theta
        phi = random.uniform(0, 2 * math.pi)  # phi
        dx = math.sin(theta) * math.cos(phi)
        dy = math.sin(theta) * math.sin(phi)
        dz = math.cos(theta)

        # O-O length
        length = 2 * bond_length

        # location of the O2
        x_O2 = x_O1 + dx * length
        y_O2 = y_O1 + dy * length
        z_O2 = z_O1 + dz * length

        # make sure point O2 is still in the simulation box
        if (0 <= x_O2 <= x_length) and (0 <= y_O2 <= y_z_length) and (0 <= z_O2 <= y_z_length):
            # location of the C
            x_C = (x_O1 + x_O2) / 2
            y_C = (y_O1 + y_O2) / 2
            z_C = (z_O1 + z_O2) / 2

            configuration.append(((x_O1, y_O1, z_O1), (x_C, y_C, z_C), (x_O2, y_O2, z_O2)))

    return configuration


#%% ########### ENERGY CALCULATION ###########
'''
Description:
This part defines the calculation of interaction energy between one selected carbon dioxide molecule and others in one configuration,
where the specific model was Gaussian-Charge Polarized model proposed by Jiang et al..
A grid-search method was used for efficiency optimization which conducted presearch for molecules within the cutoff distance before energy calculation.
Given the coordinate of the molecule selected for calculation and the configuration information,
the function "calculate_short_range_potential(molecule, configuration)" will return the corresponding interaction energy 
'''
#%% distances
# define the calculation of distance between 2 molecule
def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)

def calculate_distances_between_molecules(molecule1, molecule2):

    (O1_1, C1, O1_2) = molecule1
    (O2_1, C2, O2_2) = molecule2

    distances = {
        "O1_1_to_O2_1": distance(O1_1, O2_1),
        "O1_1_to_C2": distance(O1_1, C2),
        "O1_1_to_O2_2": distance(O1_1, O2_2),
        "C1_to_O2_1": distance(C1, O2_1),
        "C1_to_C2": distance(C1, C2),
        "C1_to_O2_2": distance(C1, O2_2),
        "O1_2_to_O2_1": distance(O1_2, O2_1),
        "O1_2_to_C2": distance(O1_2, C2),
        "O1_2_to_O2_2": distance(O1_2, O2_2),
    }

    return distances

#%% generate useful constants
SQRT_0_5 = math.sqrt(0.5)
SQRT_1_06 = math.sqrt(1.06)
SQRT_1_62 = math.sqrt(1.62)

C_C_EXP_COEF1 = 9787830
C_C_EXP_COEF2 = -3.73
C_C_LJ_COEF = 4 * 28.845
C_C_LJ_SIGMA = 2.7918
C_C_ERF_COEF = 167101 * 0.4356

C_O_EXP_COEF1 = 20125420
C_O_EXP_COEF2 = -3.827389
C_O_LJ_COEF = 4 * 48.828
C_O_LJ_SIGMA = 2.8959
C_O_ERF_COEF = 167101 * -0.2178

O_O_EXP_COEF1 = 43042140
O_O_EXP_COEF2 = -3.93
O_O_LJ_COEF = 4 * 82.656
O_O_LJ_SIGMA = 3
O_O_ERF_COEF = 167101 * 0.1089

#%% energy calculations
def erf_approx(x):
    # erf(x) ≈ 1 - (a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5) * exp(-x^2)
    # where t = 1 / (1 + p*x)
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    sign = 1 if x >= 0 else -1
    x = abs(x)

    t = 1.0 / (1.0 + p * x)
    y = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t
    return sign * (1.0 - y * math.exp(-x * x))


def get_energy(distance, indicator):  # 1 denotes CC, 2 denotes CO, 3 denotes OO
    reciprocal_distance = 1.0 / distance

    if indicator == 1:
        erf_factor = erf_approx(distance / SQRT_0_5)
        if distance > 2:
            energy = C_C_EXP_COEF1 * math.exp(C_C_EXP_COEF2 * distance) - 92009 * (reciprocal_distance ** 6) + C_C_ERF_COEF * reciprocal_distance * erf_factor
        else:
            factor = (C_C_LJ_SIGMA * reciprocal_distance) ** 6
            energy = C_C_LJ_COEF * (factor ** 2 - factor) + C_C_ERF_COEF * reciprocal_distance * erf_factor
            
        return energy
    
    if indicator == 2:
        erf_factor = erf_approx(distance / SQRT_1_06)
        if distance > 2.3:
            energy = C_O_EXP_COEF1 * math.exp(C_O_EXP_COEF2 * distance) - 131390 * (reciprocal_distance ** 6) + C_O_ERF_COEF * reciprocal_distance * erf_factor
        else:
            factor = (C_O_LJ_SIGMA * reciprocal_distance) ** 6
            energy = C_O_LJ_COEF * (factor ** 2 - factor) + C_O_ERF_COEF * reciprocal_distance * erf_factor
        
        return energy
    
    if indicator == 3:
        erf_factor = erf_approx(distance / SQRT_1_62)
        if distance > 2.7:
            energy = O_O_EXP_COEF1 * math.exp(O_O_EXP_COEF2 * distance) - 187626 * (reciprocal_distance ** 6) + O_O_ERF_COEF * reciprocal_distance * erf_factor
        else:
            factor = (O_O_LJ_SIGMA * reciprocal_distance) ** 6
            energy = O_O_LJ_COEF * (factor ** 2 - factor) + O_O_ERF_COEF * reciprocal_distance * erf_factor
        
        return energy

#%% energy lookup table generate       
def generate_lookup_table_array(func, indicator, start, end, precision):
    num_elements = int((end - start) / precision) + 1
    table = [0] * num_elements
    for i in range(num_elements):
        distance = start + i * precision
        table[i] = func(distance, indicator)
    return table

precision = 0.001
lookup_table_C_C = generate_lookup_table_array(get_energy, 1, precision, 20, precision)
lookup_table_C_O = generate_lookup_table_array(get_energy, 2, precision, 20, precision)
lookup_table_O_O = generate_lookup_table_array(get_energy, 3, precision, 20, precision)

def energy_C_C(distance_CC, start=0, precision=0.001):
    index = int((distance_CC - start) / precision)
    return lookup_table_C_C[index]

def energy_C_O(distance_CO, start=0, precision=0.001):
    index = int((distance_CO - start) / precision)
    return lookup_table_C_O[index]

def energy_O_O(distance_OO, start=0, precision=0.001):
    index = int((distance_OO - start) / precision)
    return lookup_table_O_O[index]

def calculate_potential_between_molecules(molecule1, molecule2):
    distances = calculate_distances_between_molecules(molecule1, molecule2)
    energy = energy_C_C(distances["C1_to_C2"]) + energy_C_O(distances["O1_1_to_C2"]) + energy_C_O(
        distances["C1_to_O2_1"]) + energy_C_O(distances["C1_to_O2_2"]) + energy_C_O(
        distances["O1_2_to_C2"]) + energy_O_O(distances["O1_1_to_O2_1"]) + energy_O_O(
        distances["O1_1_to_O2_2"]) + energy_O_O(distances["O1_2_to_O2_1"]) + energy_O_O(distances["O1_2_to_O2_2"])
    return energy

#%% define the V_ext
def external_potential_generate():
    x_wall = random.uniform(x_length / 20, 3 * x_length / 20)
    A = []
    phi = []
    x1 = []
    x2 = []
    V1 = []
    V2 = []
    for _ in range(4):
        A.append(T * random.normalvariate(0, 1))
        phi.append(random.uniform(0, 2 * math.pi))

    while len(x1) < 4:
        x_int = random.randint(200, 700)
        x1_test = x_int * x_length / 1000
        x2_test = random.randint(x_int + 100, 1000) * x_length / 1000

        if all(abs(x1_test - x1i) >= 2 and abs(x2_test - x2i) >= 2 and abs(x1_test - x2i) >= 2 and abs(x2_test - x1i) >= 2 for x1i, x2i in zip(x1, x2)):
            x1.append(x1_test)
            x2.append(x2_test)

    for _ in range(4):
        V1.append(T * random.normalvariate(-1, 1))
        V2.append(T * random.normalvariate(-1, 1))
    return x_wall, A, phi, x1, x2, V1, V2

def record_external_potential(filename, x_bins):
    energy = [0] * (x_bins)

    def external_potential(x):
        energy = 0

        if abs(x - x_length / 2) > (x_length / 2 - x_wall):
            energy = 100000
            return energy
        else:
            for i in range(len(A)):
                energy += A[i] * math.sin(2 * (i + 1) * math.pi * x / x_length + phi[i]) - 1.2 * abs(A[i])
        for i in range(len(V1)):
            if x1[i] < x < x2[i]:
                energy += V1[i] + (V2[i] - V1[i]) * (x - x1[i]) / (x2[i] - x1[i])
        return energy

    for i in range(x_bins):
        x = (i + 1 / 2) * x_length / x_bins
        energy[i] = external_potential(x)

    with open(filename, 'w') as file:
        for eng in energy:
            file.write(f"{eng}\n")

    return


def external_potential(positon):
    energy = 0
    x = positon[0]

    if abs(x - x_length / 2) > (x_length / 2 - x_wall):
        energy = float('inf')
        return energy
    else:
        for i in range(len(A)):
            energy += A[i] * math.sin(2 * (i + 1) * math.pi * x / x_length + phi[i]) - 1.2 * abs(A[i])

        for i in range(len(V1)):
            if x1[i] < x < x2[i]:
                energy += V1[i] + (V2[i] - V1[i]) * (x - x1[i]) / (x2[i] - x1[i])
        return energy

def calculate_external_potential(molecule):
    (O1, C, O2) = molecule
    energy = external_potential(O1) + external_potential(O2) + external_potential(C)
    return energy

#%% parameters record 
def record_parameters_to_txt(filename, *params):
    with open(filename, 'w') as f:
        for param in params:
            if isinstance(param, list):
                f.write(' '.join(map(str, param)) + '\n')
            else:
                f.write(str(param) + '\n')

def read_parameters_from_txt(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Ensure we have exactly 1+9 parameters
    if len(lines) != 10:
        raise ValueError(f"Expected 10 lines in {filename}, but found {len(lines)} lines.")

    # Parse each line into respective variables
    if lines[0].strip() == "None":
        RNG_seed = None
    else:
        RNG_seed = int(lines[0].strip())
    x_wall = float(lines[1].strip())
    A = [float(val) for val in lines[2].strip().split(' ')]
    phi = [float(val) for val in lines[3].strip().split(' ')]
    x1 = [float(val) for val in lines[4].strip().split(' ')]
    x2 = [float(val) for val in lines[5].strip().split(' ')]
    V1 = [float(val) for val in lines[6].strip().split(' ')]
    V2 = [float(val) for val in lines[7].strip().split(' ')]

    return RNG_seed ,x_wall, A, phi, x1, x2, V1, V2

#%% define the interaction energy between one molecule and all other molecules
cut_off = 10

def calculate_short_range_potential(molecule, configuration):
    """
    This function is to apply periodic boundary conditions in x, y and z directions.
    To further improve computational efficiency, a grid-searching algorithm is introduced
    to calculate the short-range interaction energies efficiently.
    """

    def grid_search(molecule_search):
        (O1, C, O2) = molecule_search
        (x, y, z) = C
        index = (int(x // (x_length / 100)) + 1, (int(y // (y_z_length / 100)) + 1),
                 (int(z // (y_z_length / 100)) + 1))  # 100 grids for each dimension
        return index

    result = []
    for i in range(len(configuration)):
        index = grid_search(configuration[i])
        result.append((i, index))

    index_cal = grid_search(molecule)
    (n_x, n_y, n_z) = index_cal
    margin_x = cut_off / (x_length / 100)
    margin_y_z = cut_off / (y_z_length / 100)

    def is_within_bounds(index, n_x, n_y, n_z, margin_x, margin_y_z):
        x, y, z = index
        return (n_x - margin_x <= x <= n_x + margin_x) and (n_y - margin_y_z <= y <= n_y + margin_y_z) and (
                n_z - margin_y_z <= z <= n_z + margin_y_z)

    filtered_result = [item for item in result if
                       is_within_bounds(item[1], n_x, n_y, n_z, margin_x, margin_y_z) and item[1] != index_cal]

    def is_within_BC_boundaries(index, n_x, n_y, n_z, ind_x, ind_y, ind_z):
        x, y, z = index

        def check_boundary(coord, n_coord, margin, status):
            if status == 1:
                return coord <= n_coord + margin - 100
            elif status == 0:
                return n_coord - margin <= coord <= n_coord + margin
            elif status == -1:
                return coord >= n_coord - margin + 100

        x_in_bounds = check_boundary(x, n_x, margin_x, ind_x)
        y_in_bounds = check_boundary(y, n_y, margin_y_z, ind_y)
        z_in_bounds = check_boundary(z, n_z, margin_y_z, ind_z)
        
        return x_in_bounds and y_in_bounds and z_in_bounds

    energy = calculate_external_potential(molecule)

    for i in range(len(filtered_result)):
        energy += calculate_potential_between_molecules(molecule, configuration[filtered_result[i][0]])
        
    def check_point_boundaries(a, b, c):
        # Helper function to determine boundary status for a single dimension
        def boundary_status(coord, margin):
            if coord < margin:
                return -1  # Close to lower boundary
            elif coord > 100 - 1 - margin:
                return 1  # Close to upper boundary
            else:
                return 0  # Not near any boundary

        # Check boundaries for x, y, z
        x_status = boundary_status(a, margin_x)
        y_status = boundary_status(b, margin_y_z)
        z_status = boundary_status(c, margin_y_z)

        return x_status, y_status, z_status
    
    x_status, y_status, z_status = check_point_boundaries(n_x, n_y, n_z)

    if x_status == 0 and y_status == 0 and z_status == 0:
        return energy
    else:
        filtered_result_new = [item for item in result if is_within_BC_boundaries(item[1], n_x, n_y, n_z, x_status, y_status, z_status)]
        for i in range(len(filtered_result_new)):
            ((x_O1, y_O1, z_O1), (x_C, y_C, z_C), (x_O2, y_O2, z_O2)) = configuration[filtered_result_new[i][0]]
            molecule_cal = (
                (x_O1 + x_status * x_length, y_O1 + y_status * y_z_length, z_O1 + z_status * y_z_length),
                (x_C + x_status * x_length, y_C + y_status * y_z_length, z_C + z_status * y_z_length),
                (x_O2 + x_status * x_length, y_O2 + y_status * y_z_length, z_O2 + z_status * y_z_length))
            energy += calculate_potential_between_molecules(molecule, molecule_cal)

        return energy

#%% ########### TRIAL MOVE OF MONTE CARLO ###########
'''
Description:
This part defines the trial move of monte carlo simulation, which contains four types of moves, namely insertion, deletion, translation, and rotation.
The acceptance possibility of each move was computed by corresponding formula in GCMC.
'''

## define the perturbation
def detect(molecule, configuration):
    def calculate_distances_between_carbon(molecule1, molecule2):
        (O1_1, C1, O1_2) = molecule1
        (O2_1, C2, O2_2) = molecule2

        def distance(point1, point2):
            return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)

        distance_C_C = distance(C1, C2)
        return distance_C_C

    for i in range(len(configuration) - 2):
        distance = calculate_distances_between_carbon(molecule, configuration[i])
        if distance < 0.5:
            return True
    return False

# define rotation (theta in x now)
def rotation(molecule):
    (O1, C, O2) = molecule

    # calculate the theta and phi
    vector_diff = (O2[0] - O1[0], O2[1] - O1[1], O2[2] - O1[2])
    length = math.sqrt(vector_diff[0] ** 2 + vector_diff[1] ** 2 + vector_diff[2] ** 2) / 2
    theta = math.acos(vector_diff[0] / (2 * length))
    phi = math.atan2(vector_diff[2], vector_diff[1])

    # update the theta and phi
    if random.choice([True, False]):
        theta_change = math.radians(random.uniform(-15, 15))
        new_theta = theta + theta_change
        new_phi = phi
    else:
        phi_change = math.radians(random.uniform(-15, 15))
        new_theta = theta
        new_phi = phi + phi_change

    new_dx = length * math.cos(new_theta)
    new_dy = length * math.sin(new_theta) * math.cos(new_phi)
    new_dz = length * math.sin(new_theta) * math.sin(new_phi)
    
    O1_new = (C[0] - new_dx, C[1] - new_dy, C[2] - new_dz)
    O2_new = (C[0] + new_dx, C[1] + new_dy, C[2] + new_dz)

    return (O1_new, C, O2_new)

# define the displace operation
def displace(molecule):
    (O1, C, O2) = molecule
    direction_vector = (O1[0] - C[0], O1[1] - C[1], O1[2] - C[2])
    max_displacement = y_z_length * (1 / 20)
    displacement = (random.uniform(-max_displacement, max_displacement), random.uniform(-max_displacement, max_displacement), random.uniform(-max_displacement, max_displacement))
    C_new = (C[0] + displacement[0], (C[1] + displacement[1]) % y_z_length, (C[2] + displacement[2]) % y_z_length)
    O1_new = (C_new[0] + direction_vector[0], C_new[1] + direction_vector[1], C_new[2] + direction_vector[2])
    O2_new = (C_new[0] - direction_vector[0], C_new[1] - direction_vector[1], C_new[2] - direction_vector[2])
    return (O1_new, C_new, O2_new)

# define the perturbation
def perform_perturbation(configuration):
    molecule_index = random.randint(0, len(configuration) - 1)
    old_molecule = configuration[molecule_index]
    old_energy = calculate_short_range_potential(old_molecule, configuration)

    if random.uniform(0, 1) < 1 / 2:
        # conduct rotation
        new_molecule = rotation(old_molecule)

        new_configuration = []
        for i, molecule in enumerate(configuration):
            if i == molecule_index:
                new_configuration.append(new_molecule)
            else:
                new_configuration.append(molecule)
    else:
        # define the displace operation
        new_molecule = displace(old_molecule)

        new_configuration = []
        for i, molecule in enumerate(configuration):
            if i == molecule_index:
                new_configuration.append(new_molecule)
            else:
                new_configuration.append(molecule)

        if detect(configuration[molecule_index], new_configuration):
            return configuration

    new_energy = calculate_short_range_potential(new_configuration[molecule_index], new_configuration)

    # calculate the probability of acceptance
    if new_energy == float('inf'):
        return configuration
    else:
        indicator = (new_energy - old_energy) / T
        if indicator < 0:
            return new_configuration
        else:
            acceptance_probability = math.exp(-indicator)

    if random.uniform(0, 1) < acceptance_probability:
        return new_configuration
    else:
        return configuration

# define exchange operation
def add(configuration):
    new_molecule = generate_molecules(x_length, y_z_length, bond_length, 1)[0]
    configuration.append(new_molecule)
    if detect(configuration[len(configuration) - 1], configuration):
        energy_add_molecule = float('inf')
        return configuration, energy_add_molecule
    else:
        energy_add_molecule = calculate_short_range_potential(new_molecule, configuration)
        return configuration, energy_add_molecule

def perform_exchange(configuration, num_insert, num_delete):
    # conduct remove or add operation with probability of 50%
    if random.choice([True, False]):
        if len(configuration) == 1:
            return configuration, num_insert, num_delete
        index_to_remove = random.randint(0, len(configuration) - 1)
        energy_removed_molecule = calculate_short_range_potential(configuration[index_to_remove], configuration)

        if energy_removed_molecule == float('inf'):
            num_delete += 1
            configuration.pop(index_to_remove)  # conduct remove operation
            return configuration, num_insert, num_delete

        indicator = (energy_removed_molecule - mu) / T
        if indicator > math.log(((x_length * y_z_length ** 2) / (LAMBDA) ** 3) / len(configuration)):
            # acceptance_probability = 1.0
            configuration.pop(index_to_remove)
            num_delete += 1
            return configuration, num_insert, num_delete
        else:
            acceptance_probability = (len(configuration) / ((x_length * y_z_length ** 2) / (LAMBDA) ** 3)) * math.exp(
                indicator)

        # determine the acceptance of new configuration
        if random.uniform(0, 1) < acceptance_probability:
            configuration.pop(index_to_remove)  # conduct remove operation
            # print("Accepted deletion.")
            num_delete += 1
            return configuration, num_insert, num_delete
        else:
            # print("Rejected deletion.")
            return configuration, num_insert, num_delete

    else:
        configuration, energy_added_molecule = add(configuration)
        # calculate the probability of acceptance
        if energy_added_molecule == float('inf'):
            configuration.pop(len(configuration) - 1)
            return configuration, num_insert, num_delete
        else:
            indicator = (mu - energy_added_molecule) / T
            if indicator > math.log((len(configuration) + 1) / ((x_length * y_z_length ** 2) / (LAMBDA) ** 3)):
                # acceptance_probability = 1
                num_insert += 1
                return configuration, num_insert, num_delete
            else:
                acceptance_probability = (((x_length * y_z_length ** 2) / (LAMBDA) ** 3) / (
                        len(configuration) + 1)) * math.exp(indicator)

        # determine the acceptance of new configuration
        if random.uniform(0, 1) < acceptance_probability:
            # print("Accepted insertion.")
            num_insert += 1
            return configuration, num_insert, num_delete
        else:
            # print("Rejected insertion.")
            configuration.pop(len(configuration) - 1)
            return configuration, num_insert, num_delete

#%% ########### GCMC AND SAMPLING ###########
'''
Description:
This part defines the density profile sampling after reaching equilibrium in a single GCMC simulation
and the details for GCMC, eg. total simulation steps, sampling steps, sampling interval and so on.
'''

#%% define density calculator
# Use C to respresent CO2, theta in x direction
def calculate_density_profile(configuration, theta_bins, phi_bins, x_bins):
    # initialization
    x_density = [0] * x_bins
    theta_density = [0] * theta_bins
    phi_density = [0] * phi_bins

    x_interval = x_length / x_bins
    theta_interval = math.pi / theta_bins
    phi_interval = 2 * math.pi / phi_bins

    for molecule in configuration:
        O1, C, O2 = molecule

        # x_C
        x_position_C = C[0]
        x_position_O1 = O1[0]
        x_position_O2 = O2[0]

        # \theta and \phi
        direction = (O2[0] - O1[0], O2[1] - O1[1], O2[2] - O1[2])
        direction_magnitude = math.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
        unit_direction = (direction[0] / direction_magnitude, direction[1] / direction_magnitude, direction[2] / direction_magnitude)
        theta = math.acos(unit_direction[0])
        phi = math.atan2(unit_direction[2], unit_direction[1])

        x_bin_C = int(x_position_C // x_interval)
        #x_bin_O1 = int(x_position_O1 // x_interval)
        #x_bin_O2 = int(x_position_O2 // x_interval)
        x_density[x_bin_C] += 1
        #x_density[x_bin_O1] += 1
        #x_density[x_bin_O2] += 1

        theta_bin = int(theta // theta_interval)
        phi_bin = int((phi + math.pi) % (2 * math.pi) // phi_interval)
        theta_density[theta_bin] += 1
        phi_density[phi_bin] += 1

    #x_density = [x / (3 * (x_length * (y_z_length ** 2) / x_bins)) for x in x_density]  # unit is set as 1/angstrom^3
    x_density = [x / (1 * (x_length * (y_z_length ** 2) / x_bins)) for x in x_density]  # unit is set as 1/angstrom^3
    theta_density = [theta / ((180 / theta_bins) * len(configuration)) for theta in theta_density]  # ~~ 1/degree (normalized)
    phi_density = [phi / ((360 / phi_bins) * len(configuration)) for phi in phi_density]  # ~~ 1/degree (normalized)
    return x_density, theta_density, phi_density

# 3D sampling
def calculate_density_profile_full(configuration, x_bins:int, theta_bins:int, phi_bins:int):
    # initialization
    # x_density = numpy.zeros((x_bins))
    shape=(x_bins,theta_bins,phi_bins)
    full_density_profile=numpy.zeros(shape)

    x_interval = x_length / x_bins
    theta_interval = math.pi / theta_bins
    phi_interval = 2 * math.pi / phi_bins

    for molecule in configuration:
        O1, C, O2 = molecule

        # x_C
        x_position_C = C[0]
        x_position_O1 = O1[0]
        x_position_O2 = O2[0]

        # \theta and \phi
        direction = (O2[0] - O1[0], O2[1] - O1[1], O2[2] - O1[2])
        direction_magnitude = math.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
        unit_direction = (direction[0] / direction_magnitude, direction[1] / direction_magnitude, direction[2] / direction_magnitude)
        theta = math.acos(unit_direction[0])
        phi = math.atan2(unit_direction[1], unit_direction[2])

        x_bin_C = int(x_position_C // x_interval)
        theta_bin = int(theta // theta_interval)
        phi_bin = int((phi + math.pi) % (2 * math.pi) // phi_interval)

        full_density_profile[x_bin_C,theta_bin,phi_bin]+=1
    return full_density_profile

#%% density save
def save_density_to_txt(x_density, theta_density, phi_density, filename):
    with open(filename, 'w') as file:
        # save x_density
        file.write("x_density\n")
        for x in x_density:
            file.write(f"{x}\n")

        # save theta_density
        file.write("\ntheta_density\n")
        for theta in theta_density:
            file.write(f"{theta}\n")

        # save phi_density
        file.write("\nphi_density\n")
        for phi in phi_density:
            file.write(f"{phi}\n")
# 3D sampling save in one commond, so no function here

# write the log into a log_file
def write_log(log_file, message):
    with open(log_file, 'w') as f:
        f.write(message + '\n')

# define the GCMC simulation
def GCMC(current_configuration, simulation_steps, simulation_output_steps, sampling_steps, sampling_interval,
         sampling_output_steps, theta_bins, phi_bins, x_bins):
    num_exchange = 0
    num_insert = 0
    num_delete = 0
    total_number = 0
    x_density = [0] * x_bins
    theta_density = [0] * theta_bins
    phi_density = [0] * phi_bins
    shape = (x_bins,theta_bins,phi_bins)
    full_density_record = numpy.zeros(shape)
    start_time = time.time()
    record_external_potential(external_potential_file, x_bins)

    for step in range(simulation_steps):
        if random.uniform(0, 1) < 0.2:
            num_exchange += 1
            current_configuration, num_insert, num_delete = perform_exchange(current_configuration, num_insert,
                                                                             num_delete)
        else:
            current_configuration = perform_perturbation(current_configuration)

        total_number += len(current_configuration)

        if step % simulation_output_steps == 0:  # output the system status every "simulation_output_steps" steps
            average_number = total_number / simulation_output_steps
            total_number = 0
            save_configuration_to_txt(current_configuration, txt_save)
            end_time = time.time()
            interval = end_time - start_time
            log_message = f"Step {step}:Number of exchange = {num_exchange}, Number of insertion = {num_insert}, Number of deletion = {num_delete}, Number of Molecules = {len(current_configuration)}, Average_number = {average_number}, Time interval = {interval:.2f}"
            print(log_message)
            write_log(log_file, log_message)
            num_exchange = 0
            num_delete = 0
            num_insert = 0

    total_number = 0  # clear the total_number
    num_exchange = 0
    num_insert = 0
    num_delete = 0

    # sampling period
    for sample_step in range(sampling_steps):
        if random.uniform(0, 1) < 0.1:
            num_exchange += 1
            current_configuration, num_insert, num_delete = perform_exchange(current_configuration, num_insert,
                                                                             num_delete)
        else:
            current_configuration = perform_perturbation(current_configuration)

        total_number += len(current_configuration)

        if sample_step % sampling_output_steps == 0:  # output the system status every "sampling_output_steps" steps
            average_number = total_number / sampling_output_steps
            total_number = 0
            end_time = time.time()
            interval = end_time - start_time
            log_message = f"Step {sample_step}:Number of exchange = {num_exchange}, Number of insertion = {num_insert}, Number of deletion = {num_delete}, Number of Molecules = {len(current_configuration)}, Average_number = {average_number}, Time interval = {interval:.2f}"
            print(log_message)
            write_log(log_file, log_message)
            save_configuration_to_txt(current_configuration, txt_save)
            num_exchange = 0
            num_delete = 0
            num_insert = 0

        if sample_step % sampling_interval == 0:
            x_density_delta, theta_density_delta, phi_density_delta = calculate_density_profile(current_configuration,
            theta_bins, phi_bins, x_bins)
            full_density_profile_delta = calculate_density_profile_full(current_configuration,
                                                    x_bins,theta_bins, phi_bins)
            x_density = [x + delta for x, delta in zip(x_density, x_density_delta)]
            theta_density = [theta + delta for theta, delta in zip(theta_density, theta_density_delta)]
            phi_density = [phi + delta for phi, delta in zip(phi_density, phi_density_delta)]
            full_density_record += full_density_profile_delta

            if sample_step > 0:
                density_profile_x = [x / (sample_step // sampling_interval) for x in x_density]
                density_profile_theta = [theta / (sample_step // sampling_interval) for theta in theta_density]
                density_profile_phi = [phi / (sample_step // sampling_interval) for phi in phi_density]
                save_density_to_txt(density_profile_x, density_profile_theta, density_profile_phi, density_save)
                full_density_profile=full_density_record/(sample_step // sampling_interval) 
                numpy.save(full_density_profile_file,full_density_profile)
                numpy.savetxt(dis_density_profile_file_txt,full_density_profile)
                numpy.savetxt(dis_density_profile_file_txt,numpy.mean(full_density_profile,axis=(1,2)))
                numpy.savetxt(ang_density_profile_file_txt,numpy.mean(full_density_profile,axis=(0)))

    save_configuration_to_txt(current_configuration, txt_save)
    return current_configuration

########### CONDUCT GCMC SIMULATION ###########
'''
if_continue = False: the parameters for external potential and a random new configuration will be automatically generated, which are both saved in two local .txt file.
if_continue = True: the parameters for external potential and the configuration will be loaded from the two exiting local .txt file 
'''
def main(RNG_seed:int|None=None,continue_folder:str|None=None):
    """
    RNG_seed:int : fixed seed in random 
    continue_folder:str : if not None, read from folder and continue GCMC
    """
    if_continue=False
    if continue_folder is not None:
        folder=continue_folder
        if_continue=True
    elif RNG_seed is not None:
        random.seed(RNG_seed)
        folder=f"record_seeded/seed{RNG_seed}"
    else:
        folder="record"
    
    os.makedirs(folder,exist_ok=True)
    paths_define(folder)
    global configuration, x_wall, A, phi, x1, x2, V1, V2
    if if_continue:
        RNG_seed_old, x_wall, A, phi, x1, x2, V1, V2 = read_parameters_from_txt(external_potential_params_file)
        configuration = read_configuration_from_txt(txt_read)
        if (RNG_seed_old is not None) and (RNG_seed is None):
            RNG_seed=RNG_seed_old
    else:
        x_wall, A, phi, x1, x2, V1, V2 = external_potential_generate()
        record_parameters_to_txt(external_potential_params_file,RNG_seed, x_wall, A, phi, x1, x2, V1, V2, beta_mu, T)
        configuration = generate_molecules(x_length, y_z_length, bond_length, num_molecule)
        save_configuration_to_txt(configuration, txt_save_0)

    save_configuration_to_txt(configuration, txt_save)
    if RNG_seed is not None:
        random.seed(RNG_seed)

    configuration = GCMC(configuration, simulation_steps = 4000000, simulation_output_steps = 1000, sampling_steps = 0, sampling_interval = 10,
                        sampling_output_steps = 50000000, theta_bins = 180, phi_bins = 360, x_bins = 1000)
    
if __name__ == "__main__":
    main()

