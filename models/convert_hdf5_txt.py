import h5py
import numpy as np
import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument('model_file', type=str, help='Path to HDF5 model file')
args = parser.parse_args()

file_path = args.model_file

# Extract the base name without the '_tf' suffix
base_name = os.path.splitext(os.path.basename(file_path))[0].replace('_tf', '')
output_dir = f"{base_name}_decomposed"
config_file = f"{base_name}_config.json"

def save_dataset_to_txt(dataset, path):
    data = dataset[:]
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(f'{path}.txt', data.flatten())

with h5py.File(file_path, 'r') as f:
    # Extract model architecture
    model_config = json.loads(f.attrs['model_config'])
    with open(config_file, 'w') as json_file:
        json.dump(model_config, json_file)

    # Extract weights and save them in a Fortran-readable format
    def recursively_save_datasets(group, path=''):
        for key in group.keys():
            item = group[key]
            new_path = os.path.join(path, key)
            if isinstance(item, h5py.Group):
                recursively_save_datasets(item, new_path)
            elif isinstance(item, h5py.Dataset):
                # Update path to include the output directory
                full_path = os.path.join(output_dir, new_path)
                save_dataset_to_txt(item, full_path)

    recursively_save_datasets(f['model_weights'])