import numpy as np
import torch
from scipy.interpolate import interp1d
import pandas as pd

def process_gaussian_data(filename, bins_hist=30):
    
    customized_gaussian_current = pd.read_csv(filename, header=None)
    customized_gaussian_current = customized_gaussian_current.values.flatten()
    # calculate mean and standard deviation
    mean_value = np.mean(customized_gaussian_current)
    sigma_value = np.std(customized_gaussian_current)

    # plt.figure(figsize=(10, 6))
    # plt.hist(customized_gaussian_current, bins=bins_hist, alpha=0.7, color='b', label="Normalized samples")
    # plt.legend()
    # plt.title(f'Normalized Gaussian Distribution (mean={mean_value:.2f}, sigma={sigma_value:.2f})')
    # plt.show()

    return customized_gaussian_current


def custom_icdf(data, uniform_random_numbers):
	original_shape = uniform_random_numbers.shape # save the original shape of the uniform random numbers
	uniform_random_numbers_flat = uniform_random_numbers.flatten() 
	#calculate the empirical cdf
	data_sorted = np.sort(data)
	data_sorted = data_sorted.reshape(-1)
	cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
		
	#inverse cdf
	inverse_cdf = interp1d(cdf, data_sorted, bounds_error=False, fill_value="extrapolate")
	generated_random_numbers = inverse_cdf(uniform_random_numbers_flat)
	generated_random_numbers = torch.from_numpy(generated_random_numbers)
	generated_random_numbers = generated_random_numbers.reshape(original_shape).to(dtype=torch.float32)
	return generated_random_numbers

def custom_sample(data, shape):
	# generate uniform random numbers
	uniform_samples = torch.rand(shape)
	# generate random numbers from the data distribution
	return custom_icdf(data, uniform_samples)