Quantum Prime Scanner
Experimental Realization of the Inverse Hilbert-Pólya Operator: Physical Sieve for Primes via Spectral Resonance Potential
This repository contains the computational engine for the paper "Experimental Realization of the Inverse Hilbert-Pólya Operator". It implements a GPU-accelerated Hamiltonian simulation that treats Riemann zeta zeros as input parameters to construct a "Spectral Resonance Potential" (SRP). The system acts as a physical sieve, where prime numbers emerge as stable resonant quantum states.
Key Findings & Capabilities
Inverse Spectral Approach: Successfully reconstructs the prime staircase and isolates prime locations using wave interference from Riemann zeros.
100% Accuracy: Achieves perfect prime detection accuracy within the theoretical resolution bounds (tested up to N=100,000).
Heisenberg-Riemann Limit: Empirically validates the physical resolution limit of the sieve. For N=2,000,000 zeros, the system exhibits a "Resonance Peak" (SNR > 13.0) exactly at the theoretical boundary (X ~ 180,000) before undergoing phase decoherence.
Requirements
To replicate the results, high-performance computing hardware is required:
Hardware: NVIDIA GPU (RTX 30xx/40xx/50xx series recommended, min. 6GB VRAM).
Software: Python 3.8+, CUDA Toolkit.
Python Dependencies
Install the required libraries via pip:



pip install numpy cupy-cuda12x matplotlib sympy tqdm


Note: Replace cupy-cuda12x with the version matching your installed CUDA toolkit (e.g., cupy-cuda11x).
Dataset (Critical)
The simulation relies on high-precision spectral data.
Download: You need the first 2,000,000 non-trivial Riemann zeros (imaginary parts). Data can be sourced from Andrew Odlyzko's tables or similar repositories.
Format: The file must be named zeros6.txt and placed in the root directory.
Format: Plain text, one floating-point number per line (e.g., 14.134725...).
Placement:
/quantum_prime_scanner
├── quantum_prime_scanner.py
├── heisenberg_limits.py
├── quantum_prime_scanner_more_numbers.py
└── zeros6.txt  <-- Place dataset here


Usage & Reproduction
1. Main Simulation (Figures 1-5)
Runs the Quantum Scanner, verifies detection accuracy (ACC), and generates the visualization of the potential, staircase, and energy spectrum.



python quantum_prime_scanner.py


Outputs:
Fig1_QuantumStaircase.png
Fig2_DerivativePeaks.png
Fig3_TwinPrimesWave.png
Fig4_SpectralResonancePotential.png
Fig5_ParticleSpectrum.png
Console output: Accuracy verification for N=30,000 and N=100,000.
2. Heisenberg Limit Stress Test (Figure 6)
Performs the large-scale scaling analysis up to X=3,000,000 to verify the resolution limit and the resonance peak.



python heisenberg_limits.py


Outputs:
Fig6_HeisenbergLimit_FullTest.png: Visualization of Signal-to-Noise Ratio (SNR) vs. Number Line Position, showing the phase transition at the theoretical limit.
3. Extended Accuracy Verification
Runs an extended verification of prime detection accuracy for larger ranges.



python quantum_prime_scanner_more_numbers.py


Outputs:
Real-time accuracy verification printed to console for larger datasets.
Updated versions of Figures 1-5 (e.g., Fig1_QuantumStaircase2.png).
Troubleshooting (OOM Errors)
If you encounter a cupy.cuda.memory.OutOfMemoryError, the batch size exceeds your GPU's VRAM capacity.
Fix: Open quantum_prime_scanner.py (or the script you are running) and modify the configuration in the compute_pi_qm method:



# Reduce tile sizes for lower VRAM usage
tile_size_x = 10000        # Default: 50000
zeros_batch_size = 1024    # Default: 2048


License
Copyright (c) 2025 Patryk Wolny.
Source Code: Licensed under the MIT License.
Article & Figures: Licensed under Creative Commons Attribution 4.0 (CC BY 4.0).
You are free to use, modify, and distribute this work, provided that original authorship is credited.
