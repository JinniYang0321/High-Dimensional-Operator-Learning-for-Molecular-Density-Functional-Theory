import numpy as np
import random
import itertools
import pickle
from sklearn.preprocessing import StandardScaler

def extract():
    data_file = 'original_data.npz'
    data = np.load(data_file)

    array1 = data['theta_density']
    array2 = data['phi_density']
    array3 = data['x_density']
    array4 = data['potential']
    array5 = data['temperature']
    array6 = data['mu']
    array7 = data['c1']

    indices = np.arange(2, 180, 6) # sampling 30 data points
    extracted_array1 = array1[:, indices]
    extracted_array2 = array2[:, indices]
    np.savez_compressed('processed_data.npz', x_density=array3, theta_density=extracted_array1, phi_density=extracted_array2, potential=array4, temperature=array5, mu=array6, c1=array7)

def data_generator(datafile, if_load):
    def apply_pbc(coord, grid_size):
        """Apply Periodic boundary condition"""
        return (coord % grid_size) / grid_size

    def generate_fourier_features(y2, y3):
        features = []
        features.extend([
            1,
            np.cos(2 * np.pi * y2),
            np.sin(2 * np.pi * y2),
            np.cos(2 * np.pi * y3), 
            np.sin(2 * np.pi * y3), 
            np.cos(2 * np.pi * y2) * np.cos(2 * np.pi * y3), 
            np.cos(2 * np.pi * y2) * np.sin(2 * np.pi * y3), 
            np.sin(2 * np.pi * y2) * np.cos(2 * np.pi * y3), 
            np.sin(2 * np.pi * y2) * np.sin(2 * np.pi * y3)
        ])
        return features
    
    # load original data
    data = np.load(datafile)
    num_samples = len(data['temperature'])
    
    # initialize dataset dictionary
    dataset = {
        'temp': [],
        'u_x1': [],      # x_density
        'u_x2_x3': [],   # ori_density
        'y1': [],
        'y2': [],        # Feature
        'y3': [],        # [y2, y3] location
        'G_u': []        
    }
    
    for i in range(num_samples):
        temp = [int(data['temperature'][i]) for _ in range(600)]
        x_density_raw = data['x_density'][i]
        theta_density_raw = data['theta_density'][i] * 180
        phi_density_raw = data['phi_density'][i] * 180
        
        array_u_x2_expanded = theta_density_raw.reshape(-1, 1)
        array_u_x3_expanded = phi_density_raw.reshape(1, -1)
        array_u_x2_x3_2d = array_u_x2_expanded * array_u_x3_expanded
        array_u_x2_x3 = array_u_x2_x3_2d.ravel()
        
        window_size = 600
        valid_range = range(5, len(x_density_raw) - window_size - 5)
        sampled_starts = random.sample(list(valid_range), 5)
        
        for start in sampled_starts:
            u_x1 = x_density_raw[start:start+window_size]
            c1_values = data['c1'][i][start:start+window_size]
            
            sampling_points = random.sample(range(150, 450), 150)
            for point in sampling_points:
                y1 = point / window_size
                G_u_rho = c1_values[point]
                
                grid_size = 30
                all_pairs = list(itertools.product(range(grid_size), repeat=2))
                sampled_pairs = random.sample(all_pairs, 20)
                
                for x2, x3 in sampled_pairs:
                    y2 = apply_pbc(x2, grid_size)
                    y3 = apply_pbc(x3, grid_size)
                    
                    fourier_features = generate_fourier_features(y2, y3)
                    G_u_theta = np.log(theta_density_raw[x2] + 1e-10)
                    G_u_phi = np.log(phi_density_raw[x3] + 1e-10)

                    dataset['temp'].append(temp)
                    dataset['u_x1'].append(u_x1)
                    dataset['u_x2_x3'].append(array_u_x2_x3)
                    dataset['y1'].append(y1)
                    dataset['y2'].append(fourier_features)
                    dataset['y3'].append([y2, y3])
                    dataset['G_u'].append(G_u_rho + G_u_theta + G_u_phi)

    dataset['temp']    = np.array(dataset['temp'])
    dataset['u_x1']    = np.stack(dataset['u_x1'], axis=0)      # shape: (N, 600)
    dataset['u_x2_x3'] = np.stack(dataset['u_x2_x3'], axis=0)   # shape: (N, L)
    dataset['y1']      = np.array(dataset['y1'])
    dataset['y2']      = np.array(dataset['y2'])
    dataset['y3']      = np.array(dataset['y3'])
    dataset['G_u']     = np.array(dataset['G_u'])

    total_size = len(dataset['y1'])
    train_indices = []
    val_indices = []
    test_indices = []
    for i in range(0, total_size, 3000):
        block = np.arange(i, min(i+3000, total_size))
        train_indices.extend(block[:2400])
        val_indices.extend(block[2400:2700])
        test_indices.extend(block[2700:3000])
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)
    
    if if_load:
        with open("scaler_u_x1.pkl", "rb") as s1:
            scaler_u_x1 = pickle.load(s1)
        with open("scaler_u_x2_x3.pkl", "rb") as s23:
            scaler_u_x2_x3 = pickle.load(s23)
    else:
        scaler_u_x1 = StandardScaler()
        scaler_u_x2_x3 = StandardScaler()

        train_u_x1 = dataset['u_x1'][train_indices]
        scaler_u_x1.fit(train_u_x1.reshape(-1, 1))
        
        train_u_x2_x3 = dataset['u_x2_x3'][train_indices]
        scaler_u_x2_x3.fit(train_u_x2_x3.reshape(-1, 1))

        with open("scaler_u_x1.pkl", "wb") as f:
            pickle.dump(scaler_u_x1, f)
        with open("scaler_u_x2_x3.pkl", "wb") as f:
            pickle.dump(scaler_u_x2_x3, f)

    def transform_data(data, scaler):
        original_shape = data.shape
        return scaler.transform(data.reshape(-1, 1)).reshape(original_shape)

    dataset['u_x1'][train_indices] = transform_data(dataset['u_x1'][train_indices], scaler_u_x1)
    dataset['u_x1'][val_indices]   = transform_data(dataset['u_x1'][val_indices], scaler_u_x1)
    dataset['u_x1'][test_indices]  = transform_data(dataset['u_x1'][test_indices], scaler_u_x1)
    
    dataset['u_x2_x3'][train_indices] = transform_data(dataset['u_x2_x3'][train_indices], scaler_u_x2_x3)
    dataset['u_x2_x3'][val_indices]   = transform_data(dataset['u_x2_x3'][val_indices], scaler_u_x2_x3)
    dataset['u_x2_x3'][test_indices]  = transform_data(dataset['u_x2_x3'][test_indices], scaler_u_x2_x3)
    
    # save dataset
    splits = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }
    
    for split_name, split_idx in splits.items():
        np.savez_compressed(
            f'dataset_{split_name}.npz',
            temp=dataset['temp'][split_idx],
            u_x1=dataset['u_x1'][split_idx],
            u_x2_x3=dataset['u_x2_x3'][split_idx],
            y1=dataset['y1'][split_idx],
            features=dataset['y2'][split_idx],
            y2_y3=dataset['y3'][split_idx],
            G_u=dataset['G_u'][split_idx]
        )

def validate_dataset(filename):
    data = np.load(filename)
    assert len(data['y1']) > 0, "avoid null dataset"
    assert not np.any(np.isnan(data['G_u'])), "NaN exists"
    assert np.all(data['y1'] >= 0) and np.all(data['y1'] <= 1), "wrong y1"
    print(f"dataset {filename} verified, including {len(data['temp'])} data")

def validate_standardization(filename, tol=1e-2):
    data = np.load(filename)
    u_x1 = data['u_x1']
    u_x2_x3 = data['u_x2_x3']
    
    mean_u_x1 = np.mean(u_x1)
    std_u_x1 = np.std(u_x1)
    
    mean_u_x2_x3 = np.mean(u_x2_x3)
    std_u_x2_x3 = np.std(u_x2_x3)
    
    print(f"Validating {filename}:")
    print(f"  u_x1  -> mean: {mean_u_x1:.4f}, std: {std_u_x1:.4f}")
    print(f"  u_x2_x3 -> mean: {mean_u_x2_x3:.4f}, std: {std_u_x2_x3:.4f}")
    
    assert abs(mean_u_x1) < tol, f"u_x1 mean {mean_u_x1} not within tolerance"
    assert abs(std_u_x1 - 1) < tol, f"u_x1 std {std_u_x1} not within tolerance"
    assert abs(mean_u_x2_x3) < tol, f"u_x2_x3 mean {mean_u_x2_x3} not within tolerance"
    assert abs(std_u_x2_x3 - 1) < tol, f"u_x2_x3 std {std_u_x2_x3} not within tolerance"
    print("Standardization check passed.")

if __name__ == '__main__':
    data_generator('processed_data.npz', if_load=False)
    validate_dataset('dataset_train.npz')
    validate_dataset('dataset_val.npz')
    validate_dataset('dataset_test.npz')

    validate_standardization('dataset_train.npz')
    validate_standardization('dataset_val.npz')
    validate_standardization('dataset_test.npz')
