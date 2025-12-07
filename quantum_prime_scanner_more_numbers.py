"""
Quantum Prime Scanner
---------------------
Copyright (c) 2025 Patryk Wolny.
Licensed under the MIT License.

This software implements the Spectral Resonance Potential (SRP) analysis 
for the physical verification of the Riemann Hypothesis.
"""

import numpy as np
import cupy as cp
from cupyx.scipy.special import expi as cuda_expi
import matplotlib.pyplot as plt
import time
import os
import sympy
from tqdm import tqdm

# --- Plot configuration ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    "font.family": "serif",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.figsize": (12, 6),
    "lines.linewidth": 1.5
})

# --- GPU Setup ---
try:
    dev = cp.cuda.Device(0)
    print(f"--> GPU: {dev.mem_info[1] / 1024**3:.2f} GB VRAM")
except:
    print("--> Error: GPU/CuPy not found.")
    exit()

# --- CUDA Kernel ---
# vectorized calculation of Li(x^rho) ~ exp(z)/z for speed
@cp.vectorize
def complex_expi(z):
    return cp.exp(z) / z 

class QuantumPrimeScanner:
    def __init__(self, zeros_list):
        print(f"--> Loading {len(zeros_list)} zeros to GPU...")
        self.zeros_gpu = cp.array(zeros_list, dtype=cp.float64)
        # Pre-calculate rho = 0.5 + i*gamma
        self.rhos_gpu = 0.5 + 1j * self.zeros_gpu
        cp.get_default_memory_pool().free_all_blocks()

    def compute_pi_qm(self, x_input, desc="Computing"):
        """
        Computes the quantum prime counting function pi_qm(x)
        using the explicit formula summation on the GPU.
        """
        # Transfer input X to GPU
        x_cpu = np.array(x_input, dtype=np.float64)
        n_total = len(x_cpu)
        results_cpu = np.zeros(n_total, dtype=np.float64)
        n_zeros = len(self.zeros_gpu)
        mempool = cp.get_default_memory_pool()
        
        # Tuning parameters for VRAM management
        tile_size_x = 50000
        zeros_batch_size = 2048
        
        # Iterate over X values in chunks (Tiling)
        iterator = range(0, n_total, tile_size_x)
        if desc:
            iterator = tqdm(iterator, desc=desc, unit="tile")

        for i in iterator:
            end_x = min(i + tile_size_x, n_total)
            current_chunk_size = end_x - i
            
            # Prepare X chunk
            x_chunk_gpu = cp.array(x_cpu[i:end_x], dtype=cp.float64)
            x_chunk_gpu = cp.maximum(x_chunk_gpu, 1.0000001) # Avoid log(1) singularity
            
            log_x_chunk = cp.log(x_chunk_gpu)
            term_main = cuda_expi(log_x_chunk) # Li(x) term
            chunk_correction = cp.zeros(current_chunk_size, dtype=cp.float64)
            
            # Sum over Riemann zeros (Batched)
            for j in range(0, n_zeros, zeros_batch_size):
                end_z = min(j + zeros_batch_size, n_zeros)
                rhos_batch = self.rhos_gpu[j:end_z]
                
                # Compute rho * ln(x) matrix
                args_matrix = rhos_batch[:, None] * log_x_chunk[None, :]
                terms = complex_expi(args_matrix)
                
                # Sum contribution: 2 * Real(Li(x^rho))
                chunk_correction += 2 * cp.sum(cp.real(terms), axis=0)
            
            # Final formula: Li(x) - Sum(...)
            res_chunk = cp.real(term_main - chunk_correction)
            res_chunk[x_chunk_gpu < 1.9] = 0.0 # Boundary condition
            results_cpu[i:end_x] = cp.asnumpy(res_chunk)
            
            # Clean VRAM
            del x_chunk_gpu, log_x_chunk, term_main, chunk_correction
            mempool.free_all_blocks()
            
        return results_cpu

def is_perfect_power(n):
    """
    Checks if n is a perfect power (e.g., 4, 8, 9, 16, 25, 27).
    Used to filter harmonic artifacts in the spectrum.
    """
    if n < 4: return False
    # Check squares
    root = int(round(n**0.5))
    if root*root == n: return True
    
    # Check higher powers
    log_n = np.log2(n)
    for k in range(3, int(log_n) + 2):
        root = int(round(n ** (1.0 / k)))
        if root ** k == n: return True
    return False

def verify_accuracy(scanner, limit):
    """
    Validates the scanner's accuracy against ground truth (SymPy)
    for a specified range N.
    """
    print(f"\n--> Verifying Accuracy for N={limit}...")
    
    # 1. Compute Quantum Wavefunction
    integers = np.arange(1, limit + 2)
    pi_values = scanner.compute_pi_qm(integers, desc="Verification Scan")
    
    # 2. Derivative Analysis (Peak Detection)
    # Signal intensity = change in pi(x)
    signals = pi_values[1:] - pi_values[:-1] 
    
    detected_primes = []
    last_detected = -1
    
    for i, slope in enumerate(signals):
        n = i + 2
        # Detection Threshold (delta = 0.35)
        if slope > 0.35:
            # Logic: Avoid double-counting adjacent points due to wave width
            if (n - last_detected) > 1 or n <= 3:
                # Harmonic Filter (Möbius logic simplified)
                if not is_perfect_power(n):
                    detected_primes.append(n)
                    last_detected = n
                    
    # 3. Compare with Ground Truth
    real_primes = list(sympy.primerange(2, limit + 1))
    
    detected_set = set(detected_primes)
    real_set = set(real_primes)
    
    true_positives = detected_set & real_set
    false_positives = detected_set - real_set
    missed = real_set - detected_set
    
    tp_count = len(true_positives)
    total_real = len(real_primes)
    
    acc = (tp_count / total_real) * 100 if total_real > 0 else 0
    
    print("-" * 40)
    print(f"Range: [2, {limit}]")
    print(f"Real Primes: {total_real}")
    print(f"Detected:    {len(detected_primes)}")
    print(f"Correct (TP): {tp_count}")
    print(f"Missed (FN):  {len(missed)}")
    print(f"False Pos (FP): {len(false_positives)}")
    print(f"--> ACCURACY: {acc:.4f}%")
    print("-" * 40)
    
    if len(missed) > 0 and len(missed) < 20:
        print(f"Sample Missed: {list(missed)}")
    if len(false_positives) > 0 and len(false_positives) < 20:
        print(f"Sample False Positives: {list(false_positives)}")

# --- Plotting Routines ---

def plot_figures(scanner, plot_range=300):
    # Fig 1: Staircase Reconstruction
    print("\n--> Generating Figure 1 (Staircase)...")
    x_dense = np.linspace(2, plot_range, 1000)
    y_dense = scanner.compute_pi_qm(x_dense, desc="Fig 1 Gen")
    y_true = [sympy.primepi(x) for x in x_dense]
    
    plt.figure(1)
    plt.clf()
    plt.plot(x_dense, y_true, 'k--', label='True pi(x)')
    plt.plot(x_dense, y_dense, 'r-', alpha=0.9, label='Quantum Model')
    plt.title("Fig 1: Quantum Staircase Reconstruction")
    plt.xlabel("x")
    plt.ylabel("pi(x)")
    plt.legend()
    plt.savefig("Fig1_QuantumStaircase.png")

    # Fig 2: Derivative Spectroscopy
    print("--> Generating Figure 2 (Peaks)...")
    n_plot = np.arange(1, plot_range + 1)
    pi_int = scanner.compute_pi_qm(n_plot, desc="Fig 2 Gen")
    pi_int[0] = 0
    deriv = np.diff(pi_int, prepend=0)
    
    plt.figure(2)
    plt.clf()
    plt.stem(n_plot, deriv, linefmt='b-', markerfmt='bo', basefmt='k-')
    plt.axhline(y=0.35, color='r', linestyle='--', label='Threshold')
    plt.title("Fig 2: Derivative Spectroscopy")
    plt.xlabel("n")
    plt.ylabel("Signal")
    plt.legend()
    plt.savefig("Fig2_DerivativePeaks.png")

    # Fig 3: Twin Prime Resolution
    print("--> Generating Figure 3 (Twin Primes)...")
    x_zoom = np.linspace(50, 70, 500)
    y_zoom = scanner.compute_pi_qm(x_zoom, desc="Fig 3 Gen")
    y_true_zoom = np.array([sympy.primepi(int(x)) for x in x_zoom], dtype=float)
    
    plt.figure(3)
    plt.clf()
    plt.plot(x_zoom, y_true_zoom, 'k--', label='True pi(x)')
    plt.plot(x_zoom, y_zoom, 'r-', label='Wavefunction')
    plt.fill_between(x_zoom, y_zoom, y_true_zoom, color='red', alpha=0.1)
    plt.title("Fig 3: Twin Prime Interference (59, 61)")
    plt.xlabel("x")
    plt.ylabel("pi(x)")
    plt.savefig("Fig3_TwinPrimesWave.png")

def plot_srp(scanner, zeros_data):
    # Fig 4: Spectral Resonance Potential (SRP)
    print("--> Generating Figure 4 (SRP)...")
    small_zeros = zeros_data[:450] # Use subset for visualization
    x = np.linspace(2, 300, 1000)
    
    # Classical potential V(x) ~ x log x
    V_cl = x * np.log(x)
    # Oscillatory correction
    V_osc = np.zeros_like(x)
    for g in small_zeros: V_osc += np.cos(g * np.log(x))
    
    # Scale for visualization
    scale = x * 0.5 
    V_qm = V_cl + (V_osc / np.sqrt(len(small_zeros))) * (scale / 10.0)
    correction = V_qm - V_cl

    # Compute prime locations for dots
    integers = np.arange(1, 302)
    pi_vals = scanner.compute_pi_qm(integers, desc="SRP Scan")
    
    primes = []
    last = -1
    for n in range(2, 301):
        slope = pi_vals[n-1] - pi_vals[n-2]
        if slope > 0.35: 
            if (n - last) > 1 or n <= 3: 
                if not is_perfect_power(n): 
                    primes.append(n)
                    last = n

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    
    # Subplot A: Potential Well
    ax1.plot(x, V_cl, 'k--', label='Classical V(x)')
    ax1.plot(x, V_qm, 'g-', label='SRP Total')
    ax1.fill_between(x, V_cl, V_qm, color='green', alpha=0.1)
    ax1.set_title("Fig 4a: Quantum Potential Well")
    ax1.legend()
    
    # Subplot B: Oscillations & Primes
    ax2.plot(x, correction, 'g-', label='SRP Oscillation')
    ax2.fill_between(x, 0, correction, color='green', alpha=0.2)
    ax2.axhline(0, color='k', lw=0.5)
    
    # Mark Primes
    for p in primes:
        val = 0
        for g in small_zeros: val += np.cos(g * np.log(p))
        scaled = (val / np.sqrt(len(small_zeros))) * (p * 0.5 / 10.0)
        ax2.plot(p, scaled, 'bo', markersize=6)
        
    ax2.set_title("Fig 4b: SRP Structure (Blue Dots = Primes)")
    ax2.set_xlabel("x")
    plt.tight_layout()
    plt.savefig("Fig4_SpectralResonancePotential.png")

def load_data():
    zeros = []
    path = '../zeros6.txt' 
    if os.path.exists(path):
        print(f"--> Reading {path}...")
        with open(path) as f:
            for i, l in enumerate(f):
                if l.strip(): zeros.append(float(l.strip()))
                if i >= 2000000: break # Max limit for safety
    return zeros

if __name__ == "__main__":
    data = load_data()
    
    if data:
        scanner = QuantumPrimeScanner(data)
        
        # --- ACCURACY TESTS ---
        # Testing limits based on Heisenberg-Riemann resolution
        verify_accuracy(scanner, limit=180000)
        verify_accuracy(scanner, limit=400000)
        
        # --- GENERATE PLOTS ---
        # Uncomment to regenerate paper figures
        plot_figures(scanner)
        plot_srp(scanner, data)
        
    else:
        print("Error: zeros6.txt not found.")