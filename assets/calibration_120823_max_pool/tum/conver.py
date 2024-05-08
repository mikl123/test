import torch
import numpy as np

def convert_pt_to_npz(pt_file_path, npz_file_path):
    # Load data from .pt file
    data = torch.load(pt_file_path)

    # Convert to NumPy array
    numpy_array = data.numpy()

    # Save as .npz
    np.savez(npz_file_path, data=numpy_array)

# Example usage:
convert_pt_to_npz('data.pt', 'data.npz')
