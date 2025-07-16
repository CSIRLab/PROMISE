# =============================================================================
# core_function.py — Simulation core logic for PROMISE Simulator
# Author:      Emilie Ye
# Date:        2025-06-27
#
# Description:
#   This file implements the simulation backend for the PROMISE Simulator.
#   It loads NN and BNN weight files, performs probabilistic sampling using either
#   device data or ideal sources, generates plots (histograms, QQ plots, 1D/2D/3D histograms),
#   and exports simulation configurations. 
#   Supports "Demo", "Memory", and "Compute In Memory" workflows.
#
# Copyright (c) 2025
# =============================================================================

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import yaml
import torch 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from simulator.memory import *
from interface.load_parameter import *
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(out_dir, exist_ok=True)

class SimulatorCore:
    def __init__(self):
        #self.output_folder = "output_plots"

        self.weights_folder = None
        self.bnn_weights_folder = None
        self.device_data_path = None
        self.sampled_fc1_plot = None

        self.mode = "Memory"
        self.weight = "A"
        self.technology = "65"
        self.frequency = "1e9"
        self.precision_mu = "6"
        self.precision_sigma = "2"
        self.sampling_times = "100"

        self.fc1_index = "0"
        self.fc2_index = "0"
        self.fc3_index = "0"

        self.bins = "100"

        #demo
        self.demo_mu = 0.0
        self.demo_sigma = 1.0
        self.demo_samples = 100
        self.demo_bins = 50
        self.demo_data = None # For 3d data


    def _try_load_pth(self):
    
        if not self.weights_folder:
            return None

        pth_files = [f for f in os.listdir(self.weights_folder)
                 if f.lower().endswith((".pth", ".pt"))]
        if not pth_files:
            return None

        pth_path = os.path.join(self.weights_folder, pth_files[0])
        print(f"Found checkpoint: {pth_path}")

        try:
            ckpt = torch.load(pth_path, map_location="cpu")

            state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

            fc1_w = state["fc1.weight"].cpu().numpy().flatten()
            fc2_w = state["fc2.weight"].cpu().numpy().flatten()
            fc3_w = state["fc3.weight"].cpu().numpy().flatten()

            return fc1_w, fc2_w, fc3_w
        except (KeyError, RuntimeError, ValueError) as e:
            print("Failed to load .pth parameters:", e)
            return None

    def generate_nn_weight_plots(self):
        if not self.weights_folder:
            print("No NN Weights Folder uploaded.")
            return False

        try:
            fc1_weights = np.loadtxt(os.path.join(self.weights_folder, "fc1_weights.csv"), delimiter=",")
            fc2_weights = np.loadtxt(os.path.join(self.weights_folder, "fc2_weights.csv"), delimiter=",")
            fc3_weights = np.loadtxt(os.path.join(self.weights_folder, "fc3_weights.csv"), delimiter=",")

            bins = int(self.bins)

            self.save_plot("fc1_weights_plot.svg", fc1_weights, mode="hist", bins=bins, xlabel="Weight Value", ylabel="Count")
            self.save_plot("fc2_weights_plot.svg", fc2_weights, mode="hist", bins=bins, xlabel="Weight Value", ylabel="Count")
            self.save_plot("fc3_weights_plot.svg", fc3_weights, mode="hist", bins=bins, xlabel="Weight Value", ylabel="Count")

            print("Weights plots updated.")
            return True

        except (FileNotFoundError, OSError):
            csv_loaded = False

        if not csv_loaded:
            res = self._try_load_pth()
            if res is None:
                print("Neither CSV nor .pth weights found.")
                return False
            fc1_weights, fc2_weights, fc3_weights = res
            bins = int(self.bins)

            self.save_plot("fc1_weights_plot.svg", fc1_weights, mode="hist", bins=bins, xlabel="Weight Value", ylabel="Count")
            self.save_plot("fc2_weights_plot.svg", fc2_weights, mode="hist", bins=bins, xlabel="Weight Value", ylabel="Count")
            self.save_plot("fc3_weights_plot.svg", fc3_weights, mode="hist", bins=bins, xlabel="Weight Value", ylabel="Count")

            print("Weights plots updated.")
            return True

    def _load_bnn_layer_arrays(self, fc_name):
        csv_miu   = os.path.join(self.bnn_weights_folder, f"{fc_name}_miu_w.csv")
        csv_sigma = os.path.join(self.bnn_weights_folder, f"{fc_name}_sigma_w.csv")
        if os.path.exists(csv_miu) and os.path.exists(csv_sigma):
            miu_w   = np.loadtxt(csv_miu,   delimiter=",")
            sigma_w = np.loadtxt(csv_sigma, delimiter=",")
            return miu_w, sigma_w

        pth_files = [f for f in os.listdir(self.bnn_weights_folder)
                 if f.lower().endswith((".pth", ".pt"))]
        if not pth_files:
            raise FileNotFoundError("No CSV or .pth found for layer " + fc_name)

        pth_path = os.path.join(self.bnn_weights_folder, pth_files[0])
        ckpt     = torch.load(pth_path, map_location="cpu", weights_only=True)
        state    = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

        key_map = {
            "fc1": ("linear1.weights_mean", "linear1.lweights_sigma"),
            "fc2": ("linear2.weights_mean", "linear2.lweights_sigma"),
            "fc3": ("linear3.weights_mean", "linear3.lweights_sigma"),
        }

        miu_key, sigma_key = key_map[fc_name]
        miu_w   = state[miu_key].cpu().numpy()
        sigma_w = state[sigma_key].cpu().numpy()

        if miu_key not in state or sigma_key not in state:
            raise KeyError(f"Keys '{miu_key}' / '{sigma_key}' not found in {pth_path}")

        miu_w   = state[miu_key].cpu().numpy()
        sigma_w = state[sigma_key].cpu().numpy()
        sigma_w = np.exp(sigma_w)
        if miu_w.ndim == 1:          # e.g. (840,)
            miu_w   = miu_w.reshape((1, -1))      # → (1, 840)
            sigma_w = sigma_w.reshape((1, -1))

        return miu_w, sigma_w


    def _get_weights_per_neuron(self, layer_name):
        # First, try to load from CSV
        try:
            filepath = os.path.join(self.bnn_weights_folder, f"{layer_name}_weights.csv")
            weights = np.loadtxt(filepath, delimiter=",")
            if weights.ndim != 2:
                raise ValueError(f"{layer_name}_weights.csv is not a 2D array.")
            return weights.shape[1]
        except Exception as e:
            print(f"CSV loading failed for {layer_name}: {e}")

        # Fallback to .pth
        try:
            pth_files = [f for f in os.listdir(self.bnn_weights_folder) if f.endswith((".pt", ".pth"))]
            if not pth_files:
                raise FileNotFoundError("No .pth files found.")

            pth_path = os.path.join(self.bnn_weights_folder, pth_files[0])
            ckpt = torch.load(pth_path, map_location="cpu")
            state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

            key_map = {
                "fc1": "linear1.weights_mean",
                "fc2": "linear2.weights_mean",
                "fc3": "linear3.weights_mean",
            }

            weight_key = key_map[layer_name]
            weight_tensor = state[weight_key]
            if weight_tensor.ndim != 2:
                raise ValueError(f"{weight_key} tensor is not 2D")

            return weight_tensor.shape[1]  # ← this is weights_per_neuron

        except Exception as e:
            print(f"Failed to extract weights_per_neuron from .pth for {layer_name}: {e}")
            return None


    def generate_bnn_weight_plots(self):
        if not self.bnn_weights_folder:
            print("No BNN Weights Folder uploaded.")
            return False

        try:
            for layer_name, index_str in [("fc1", self.fc1_index), ("fc2", self.fc2_index), ("fc3", self.fc3_index)]:
                weights_per_neuron = self._get_weights_per_neuron(layer_name)
                if weights_per_neuron is not None:
                    self._generate_single_bnn_plot(layer_name, weights_per_neuron, int(index_str))
                else:
                    print(f"Skipping {layer_name} due to error.")
            
            print("BNN weight plot updated.")
            return True

        except Exception as e:
            print("Error during BNN plotting:", e)
            return False

    def _generate_single_bnn_plot(self, fc_name, weights_per_neuron, flat_index):
        miu_w, sigma_w = self._load_bnn_layer_arrays(fc_name)

        # miu_w = np.loadtxt(os.path.join(self.bnn_weights_folder, f"{fc_name}_miu_w.csv"), delimiter=",")
        # sigma_w = np.loadtxt(os.path.join(self.bnn_weights_folder, f"{fc_name}_sigma_w.csv"), delimiter=",")

        if miu_w.ndim == 1:
            if flat_index >= miu_w.size:
                raise ValueError(f"flat_index {flat_index} ≥ {miu_w.size} (1-D length)")
            mu    = miu_w[flat_index]
            sigma = sigma_w[flat_index]
            neuron_idx = 0 
            weight_idx = flat_index
            title_suffix = f"1-D vector, Index {flat_index}"
        else:
            neuron_idx = flat_index // weights_per_neuron
            weight_idx = flat_index % weights_per_neuron

            mu = miu_w[neuron_idx, weight_idx]
            sigma = sigma_w[neuron_idx, weight_idx]

        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)
        pdf = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

        bins = int(self.bins)

        self.save_plot(f"bnn_{fc_name}_weight_plot.svg", x, y=pdf, mode="line",
                       xlabel="Weight Value", ylabel="Probability Density",
                       with_title=True,
                       title=f"BNN {fc_name.upper()} Neuron {neuron_idx}, Weight {weight_idx} (Index {flat_index})")


        self.save_qq_plot(f"bnn_{fc_name}_qq_plot.svg", mu, sigma)

        self.save_plot(f"bnn_{fc_name}_miu_dist.svg", miu_w.flatten(), mode="hist", bins=bins,
                       xlabel="μ Value", ylabel="Count", with_title=True, title=f"BNN {fc_name.upper()} μ Distribution")

        self.save_plot(f"bnn_{fc_name}_sigma_dist.svg", sigma_w.flatten(), mode="hist", bins=bins,
                       xlabel="σ Value", ylabel="Count", with_title=True, title=f"BNN {fc_name.upper()} σ Distribution")

    def generate_sampled_fc1_plots(self):
        success = self.generate_sampled_fc1_weight_plot()
        if success:
            print("BNN Sampled FC1 plots updated.")
        return success


    def generate_sampled_fc1_weight_plot(self):
        if not self.bnn_weights_folder or not self.device_data_path:
            print("Missing BNN folder or sampled weights file.")
            return False

        try:
            distribution = process_gaussian_data(self.device_data_path)

            miu_w, sigma_w = self._load_bnn_layer_arrays("fc1")
            if miu_w.ndim == 2:
                num_neurons, weights_per_neuron = miu_w.shape
            elif miu_w.ndim == 1:
                num_neurons = len(miu_w)
                weights_per_neuron = 1
            else:
                print("Unexpected BNN weight array shape.")
                return False

            sample_times = int(self.sampling_times)
            sampling_shape = (num_neurons, weights_per_neuron, sample_times)

            sampled_tensor = custom_sample(distribution, sampling_shape)
            sampled_np = sampled_tensor.cpu().numpy().reshape(-1, sample_times)

            fc1_index = int(self.fc1_index)
            sampled_data = sampled_np[fc1_index, :]


            # Calculate μ1 and σ1 from sampled_data → N1 ~ N(μ1, σ1)
            mu1 = np.mean(sampled_data)
            sigma1 = np.std(sampled_data)

            print(f"Sampled N1 mean={mu1}, std={sigma1}")

            # miu_w = np.loadtxt(os.path.join(self.bnn_weights_folder, "fc1_miu_w.csv"), delimiter=",")
            # sigma_w = np.loadtxt(os.path.join(self.bnn_weights_folder, "fc1_sigma_w.csv"), delimiter=",")
            print("μ shape =", miu_w.shape)

            if miu_w.ndim == 1:
                mu = miu_w[fc1_index]
                sigma = sigma_w[fc1_index]
            else:
                #weights_per_neuron = 400
                if weights_per_neuron is None:
                    print("Could not determine weights_per_neuron for FC1.")
                    return False


                neuron_idx = fc1_index // weights_per_neuron
                weight_idx = fc1_index % weights_per_neuron

                mu = miu_w[neuron_idx, weight_idx]
                sigma = sigma_w[neuron_idx, weight_idx]

            print(f"BNN mu={mu}, sigma={sigma}")

            # Transform sampled_data → miu + sigma * standard_N1
            standard_N1 = (sampled_data - mu1) / sigma1
            transformed_data = mu + sigma * standard_N1

            print("standard_N1 mean:", np.mean(standard_N1))
            print("standard_N1 std:", np.std(standard_N1))

            
            bins = int(self.bins)

            os.makedirs(out_dir, exist_ok=True)

            self.save_plot(
                "bnn_fc1_sampled_N1_distribution.svg",
                standard_N1,
                mode="hist",
                bins=bins,
                xlabel="Sampled N1 Value",
                ylabel="Count",
                with_title=True,
                title=f"BNN FC1 Index {fc1_index} N1 Distribution"
            )
            #miu + sigma * N1

            # Plot only histogram of transformed_data
            self.save_plot(
                "bnn_fc1_sampled_weight_plot.svg",
                transformed_data,
                mode="hist",
                bins=bins,
                xlabel="Weight Value",
                ylabel="Count",
                with_title=True,
                title=f"mu + sigma * N1 Distribution (FC1 Index {fc1_index})\nmu={mu:.3f}, sigma={sigma:.3f}"
            )
            print("Saved N1 Distribution to bnn_fc1_sampled_N1_distribution.svg")

            # Save QQ plot of transformed_data
            print("Calling save_empirical_qq_plot...")
            #out_path = os.path.join(out_dir, "bnn_fc1_sampled_weight_qq_plot.svg")

            out_path = "bnn_fc1_sampled_weight_qq_plot.svg"
            self.save_empirical_qq_plot(out_path, transformed_data, mu, sigma)

            print("BNN FC1 sampled weight plot generated.")
            self.sampled_fc1_plot = True

            return True

        except Exception as e:
            print("Error during BNN sampled weight plotting:", e)
            return False


    

    def save_plot(self, filename, x, y=None, xlabel="", ylabel="", dpi=400, mode="line", bins=100, with_title=False, title=""):
        filename = os.path.join(out_dir, filename)

        if filename.endswith(".svg"):
            figsize = (12, 8)   # Larger size for SVG
        else:
            figsize = (6, 4)    # PNG can stay smaller with high dpi

        plt.figure(figsize=figsize)

        if mode == "line":
            plt.plot(x, y, linewidth=4)
        elif mode == "hist":
            sns.histplot(x, kde=True, bins=bins, linewidth=2) #, linewidth=0.5, color='blue'

        if with_title:
            plt.title(title, fontsize=28, fontweight='bold')

        plt.xlabel(xlabel,fontsize=24, fontweight='bold')
        plt.ylabel(ylabel,fontsize=24, fontweight='bold')
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(filename, dpi=dpi)
        plt.close()


    def save_empirical_qq_plot(self, filename, data, mu, sigma, dpi=400):
        filename = os.path.join(out_dir, filename)
        print("Saving QQ plot:", os.path.abspath(filename))
        # QQ plot based on transformed data
        plt.figure(figsize=(12, 8))

        (osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm", sparams=(mu, sigma), plot=plt)

        plt.title(f"QQ Plot (BNN FC1 Index {self.fc1_index}, r={r:.4f})", fontsize=28, fontweight='bold')
        plt.xlabel("Theoretical Quantiles", fontsize=24, fontweight='bold')
        plt.ylabel("Sample Quantiles", fontsize=24, fontweight='bold')
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(filename, dpi=dpi)
        plt.close()

    def generate_nd_histogram(self, data, dim,filename, interactive=False):
        if filename is not None:
            filename = os.path.join(out_dir, filename)
            print("Saving nd_histogram:", os.path.abspath(filename))

        fig = None
        ax = None

        if dim == 1:
            fig = plt.figure(figsize=(7, 5))
        elif dim == 2:
            fig, ax = plt.subplots(figsize=(7, 5))
        elif dim == 3:
            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111, projection='3d')
        else:
            print("Unsupported dimension:", dim)
            return
            
        if dim == 1:
            plt.hist(data, bins=30, color='steelblue', edgecolor='black', linewidth=0.8)
            plt.xlabel("Value", fontsize=12)
            plt.ylabel("Count", fontsize=12)
            plt.title("1D Histogram", fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.6)

        elif dim == 2:
            if data.ndim == 1 or data.shape[1] < 2:
                print("Need at least 2D data for 2D histogram.")
                return
            x, y = data[:, 0], data[:, 1]
            cmap = plt.cm.Blues
            cmap.set_under(color='white')
            
            min_x, max_x = np.min(x), np.max(x)
            min_y, max_y = np.min(y), np.max(y)
            range_x = max_x - min_x
            range_y = max_y - min_y
            max_range = max(range_x, range_y)

            x_mid = 0.5 * (min_x + max_x)
            y_mid = 0.5 * (min_y + max_y)
            x_min = x_mid - max_range / 2
            x_max = x_mid + max_range / 2
            y_min = y_mid - max_range / 2
            y_max = y_mid + max_range / 2

            counts, xedges, yedges, im = ax.hist2d(
                x, y, bins=60, cmap=cmap, vmin=1,
                range=[[x_min, x_max], [y_min, y_max]]
            )
            vmax = counts.max()
            im.set_clim(vmin=1, vmax=vmax)

            ax.set_xlabel("X", fontsize=12)
            ax.set_ylabel("Y", fontsize=12, rotation=0, labelpad=20)
            ax.set_title("2D Histogram", fontsize=14)
            fig.colorbar(im, ax=ax, label="Frequency")
            ax.grid(True, linestyle='--', alpha=0.4)

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect('equal')

        elif dim == 3:
            if data.ndim == 1 or data.shape[1] < 3:
                print("Need at least 3D data for 3D histogram.")
                return

            #ax = fig.add_subplot(111, projection='3d')
            x, y, z = data[:, 0], data[:, 1], data[:, 2]

            # Histogram binning
            min_x, max_x = np.min(x), np.max(x)
            min_y, max_y = np.min(y), np.max(y)
            min_z, max_z = np.min(z), np.max(z)

            range_x = max_x - min_x
            range_y = max_y - min_y
            range_z = max_z - min_z
            max_range = max(range_x, range_y, range_z)

            x_mid = 0.5 * (min_x + max_x)
            y_mid = 0.5 * (min_y + max_y)
            z_mid = 0.5 * (min_z + max_z)

            x_min = x_mid - max_range / 2
            x_max = x_mid + max_range / 2
            y_min = y_mid - max_range / 2
            y_max = y_mid + max_range / 2
            z_min = z_mid - max_range / 2
            z_max = z_mid + max_range / 2

            hist, edges = np.histogramdd(
                (x, y, z),
                bins=15,
                range=[[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            )

            x_edges, y_edges, z_edges = edges

            # Bin centers
            x_centers = (x_edges[:-1] + x_edges[1:]) / 2
            y_centers = (y_edges[:-1] + y_edges[1:]) / 2
            z_centers = (z_edges[:-1] + z_edges[1:]) / 2

            # Create grid of bin centers
            Xc, Yc, Zc = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')
            values = hist.flatten()
            Xc = Xc.flatten()
            Yc = Yc.flatten()
            Zc = Zc.flatten()

            # Filter non-zero bins
            nonzero = values > 0
            Xc, Yc, Zc, values = Xc[nonzero], Yc[nonzero], Zc[nonzero], values[nonzero]

            # Normalize for color
            norm_values = values / np.max(values)
            colors = plt.cm.viridis(norm_values)

            blues = colormaps['Blues']
            newcolors = blues(np.linspace(0.12, 1, 256))
            custom_blues = ListedColormap(newcolors)
            # 3D scatter with color
            sc = ax.scatter(Xc, Yc, Zc, c=norm_values, cmap=custom_blues, s=8, alpha=0.7)
            # Labels
            ax.set_xlabel("X", fontsize=10)
            ax.set_ylabel("Y", fontsize=10)
            ax.set_zlabel("Z", fontsize=10)
            ax.set_title("3D Density Histogram", fontsize=13)

            ax.view_init(elev=25, azim=135)
            ax.set_box_aspect([1, 1, 1])
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)

            arrow_length = max_range * 0.15
            arrow_origin_x = x_min
            arrow_origin_y = y_min
            arrow_origin_z = z_min

            ax.quiver(arrow_origin_x, arrow_origin_y, arrow_origin_z, arrow_length, 0, 0, color='r', linewidth=1.5)
            ax.quiver(arrow_origin_x, arrow_origin_y, arrow_origin_z, 0, arrow_length, 0, color='g', linewidth=1.5)
            ax.quiver(arrow_origin_x, arrow_origin_y, arrow_origin_z, 0, 0, arrow_length, color='b', linewidth=1.5)

            ax.text2D(1.02, 0.95, "X → Red", transform=ax.transAxes, fontsize=10, va='top', ha='left', color='red')
            ax.text2D(1.02, 0.90, "Y → Green", transform=ax.transAxes, fontsize=10, va='top', ha='left', color='green')
            ax.text2D(1.02, 0.85, "Z → Blue", transform=ax.transAxes, fontsize=10, va='top', ha='left', color='blue')

            x_range = np.max(x) - np.min(x)
            y_range = np.max(y) - np.min(y)
            z_range = np.max(z) - np.min(z)
            max_range = max(x_range, y_range, z_range)
            #ax.set_box_aspect([x_range / max_range, y_range / max_range, z_range / max_range])
            
            fig.subplots_adjust(left=0.05, right=0.8)
            fig.colorbar(sc, ax=ax, shrink=0.6, label="Frequency", pad=0.08)
        else:
            print("Unsupported dimension:", dim)
            return

        if interactive and dim == 3:
            plt.show()
        else:
            fig.tight_layout()
            fig.savefig(filename, format='svg')
            plt.close(fig)

    def save_qq_plot(self, filename, mu, sigma, sample_size=1000, dpi=400):
        filename = os.path.join(out_dir, filename)
        
        # Generate normal distributed samples based on mu, sigma
        samples = np.random.normal(loc=mu, scale=sigma, size=sample_size)

        # QQ plot
        plt.figure(figsize=(12, 8))
        (osm, osr), (slope, intercept, r) = stats.probplot(samples, dist="norm", sparams=(mu, sigma), plot=plt)

        plt.title(f"QQ Plot (mu={mu:.3f}, sigma={sigma:.3f}, r={r:.4f})", fontsize=28, fontweight='bold')
        plt.xlabel("Theoretical Quantiles", fontsize=24, fontweight='bold')
        plt.ylabel("Sample Quantiles", fontsize=24, fontweight='bold')
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(filename, dpi=dpi)
        plt.close()

    def export_memory_config_to_yaml(self, filename="simulation_config.yaml"):
        filename = os.path.join(out_dir, filename)
        
        config = {
            "mode": "Memory",
            "device_data_file": self.device_data_path or "",
            "weight": self.weight,
            "technology": self.technology,
            "frequency": self.frequency,
            "precision_mu": self.precision_mu,
            "precision_sigma": self.precision_sigma,
            "sampling_times": self.sampling_times,
            "nn_weights_folder": self.weights_folder or "",
            "bnn_weights_folder": self.bnn_weights_folder or "",
            "bnn_fc1_index": self.fc1_index,
            "bnn_fc2_index": self.fc2_index,
            "bnn_fc3_index": self.fc3_index,
            "pie_config": self.pie_config_path or "",
            # ADC
            "adv_adc_type": getattr(self, "adc_type_value", "SAR"),
            "config_adc_config_mode":   getattr(self, "config_adc_config_mode", ""),
            "adc_yaml": getattr(self, "memory_adc_config", ""),

            # Buffer
            "config_buffer_config_mode":getattr(self, "config_buffer_config_mode", ""),
            "config_buffer_yaml":       getattr(self, "config_buffer_yaml", ""),

            # SenseAMP
            "config_senseamp_mode":     getattr(self, "config_senseamp_mode", ""),
            "config_senseamp_yaml":     getattr(self, "config_senseamp_yaml", ""),

            # Decoder
            "config_decoder_mode":      getattr(self, "config_decoder_mode", ""),
            "config_decoder_yaml":      getattr(self, "config_decoder_yaml", ""),

            # Memory tech
            "config_memory_tech":       getattr(self, "config_memory_tech", ""),
            "config_memory_config_mode":getattr(self, "config_memory_config_mode", ""),
            "config_memory_yaml":       getattr(self, "config_memory_yaml", ""),

            # Technology node (65/45/28/14 nm)
            "config_technology_node":   getattr(self, "config_technology_node", ""),
        }

        try:
            with open(filename, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            print(f"Exported config to {filename}")
        except Exception as e:
            print(f"Error exporting config to {filename}: {e}")

    def generate_all_pie_charts(self, yaml_path=None):
        if yaml_path and os.path.exists(yaml_path):
            with open(yaml_path, "r") as f:
                try:
                    pie_data = yaml.safe_load(f)
                    pie_configs = pie_data.get("pie_charts", [])
                except Exception as e:
                    print("Failed to load pie config from YAML:", e)
                    return
        else:
            print("No YAML file provided. Skipping pie chart generation.")
            return

        colors = ['#4f81bd', '#6faad4', '#9dc3e6', '#dbe5f1']
        for config in pie_configs:
            out_path = os.path.join(out_dir, config["filename"])

            fig, ax = plt.subplots(figsize=(5, 5))
            wedges, texts, autotexts = ax.pie(config["values"], autopct='%1d%%', startangle=90,colors=colors)
            for text in texts:
                text.set_fontsize(12)
                text.set_fontweight('bold')
            for autotext in autotexts:
                autotext.set_fontsize(12)
                autotext.set_fontweight('bold')
            ax.axis('equal')
            ax.set_position([0.1, 0.1, 0.8, 0.8])  
            ax.set_title(config["title"], fontsize=18, fontweight='bold')
            ax.legend(config["labels"], loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=10)
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close()

    def run_demo(self):
        mus    = getattr(self, 'demo_mus', [1.0])
        sigmas = getattr(self, 'demo_sigmas', [1.0])
        count = len(mus)
        all_demo_data = []
        epsilons = []
        for i in range(count):
            mu_i    = mus[i]
            sigma_i = sigmas[i]
            print(f"mu{i+1} = {mus[i]:.4f}, sigma{i+1} = {sigmas[i]:.4f}")


            if self.demo_mode == "ideal":
                # Standard Normal sampling
                epsilon = np.random.normal(loc=0, scale=1, size=self.demo_samples)
            else:
                all_data = np.loadtxt(self.demo_file, delimiter=",")
                epsilon_original = custom_sample(all_data, shape=(self.demo_samples,)).numpy()
                
                # Normalize
                mu_st = np.mean(epsilon_original)
                sigma_st = np.std(epsilon_original)
                epsilon = (epsilon_original - mu_st) / sigma_st

            epsilons.append(epsilon)
            print(f"epsilon_{i+1} = {epsilon}")

            demo_data = mu_i + sigma_i * epsilon
            all_demo_data.append(demo_data.reshape(-1, 1))
            hist_fname = f"demo_histogram_{i+1}.svg"
            qq_fname   = f"demo_qqplot_{i+1}.svg"

            self.save_plot(
            hist_fname,
            demo_data,
            mode="hist",
            bins=self.demo_bins,
            xlabel="Value",
            ylabel="Count",
            with_title=True,
            title=f"Demo ({self.demo_mode}) Histogram #{i+1}\nμ={mu_i:.2f}, σ={sigma_i:.2f}"
            )

            self.save_empirical_qq_plot(
                qq_fname,
                demo_data,
                mu_i,
                sigma_i
            )

        distribution_mode = getattr(self, "gaussian_mode", "Single Gaussian")

        if distribution_mode == "Single Gaussian":
            nd_hist_fname = f"demo_nd_histogram.svg"
            final_data = np.hstack(all_demo_data)  # Shape: (samples, dim)
            self.demo_data = final_data
            self.generate_nd_histogram(final_data, count, nd_hist_fname)
        else:
            alphas = getattr(self, 'demo_alphas', [1.0 / count] * count)
            if len(alphas) != count:
                raise ValueError("Number of alphas must match number of mus/sigmas.")

            # Sampling component index per alpha (Bernoulli if 2, Categorical if >2)
            k_choices = np.random.choice(count, size=self.demo_samples, p=alphas)

            # GMM sampling per mask
            mixed_data = np.zeros(self.demo_samples)
            for i in range(self.demo_samples):
                k = k_choices[i]
                # Use corresponding epsilon
                eps = np.random.normal() if self.demo_mode == "ideal" else epsilons[k][i]
                mixed_data[i] = mus[k] + sigmas[k] * eps

            self.save_plot(
                "demo_mixed_histogram.svg",
                mixed_data,
                mode="hist",
                bins=self.demo_bins,
                xlabel="Value",
                ylabel="Count",
                with_title=True,
                title=f"Gaussian Mixture Model Histogram (α = {alphas})"
            )

    def export_demo_config_to_yaml(self, filename="simulation_config.yaml"):
        filename = os.path.join(out_dir, filename)
        
        config = {
            "mode": "Demo",
            "rng_source": self.demo_mode,
            "rng_file": self.demo_file or "",
            "weight_number": len(getattr(self, "demo_mus", [])),
            "mus": getattr(self, "demo_mus", []),
            "sigmas": getattr(self, "demo_sigmas", []),
            "sampling_times": int(self.demo_samples),
            "bins": int(self.demo_bins)
        }
        try:
            with open(filename, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            print(f"Exported config to {filename}")
        except Exception as e:
            print(f"Error exporting config to {filename}: {e}")


    def reset(self):
        # should be reset between modes
        self.weights_folder = None
        self.bnn_weights_folder = None
        self.device_data_path = None
        self.pie_config_path = None

        self.weight = None
        self.technology = None
        self.frequency = None
        self.precision_mu = None
        self.precision_sigma = None
        self.sampling_times = None
        self.fc1_index = None
        self.fc2_index = None
        self.fc3_index = None
        self.bins = None
        self.architecture = None

        self.demo_mode = None
        self.demo_file = None
        self.demo_mus = None
        self.demo_sigmas = None
        self.demo_samples = None
        self.demo_bins = None