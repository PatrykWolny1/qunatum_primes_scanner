import numpy as np
import matplotlib.pyplot as plt
import sympy
import time
import os
from quantum_prime_scanner import QuantumPrimeScanner, is_perfect_power

# --- KONFIGURACJA ---
ZEROS_FILE = '../zeros6.txt'
MAX_ZEROS = 2000000  # Pełna moc: 2 miliony
# Punkty testowe (zagęszczone wokół 180,000)
TARGETS = [
    300, 1000, 5000, 10000, 30000, 50000, 100000, 
    150000, 170000, 180000, 190000, 200000, # Okolice limitu
    250000, 300000, 500000, 1000000, 3000000
]
# --------------------

def load_data_full():
    print(f"--> Loading dataset ({MAX_ZEROS} zeros)...")
    zeros = []
    if os.path.exists(ZEROS_FILE):
        with open(ZEROS_FILE) as f:
            for i, l in enumerate(f):
                if l.strip(): zeros.append(float(l.strip()))
                if i >= MAX_ZEROS - 1: break
    return np.array(zeros)

def measure_snr(pi_qm, x_range):
    """
    Oblicza SNR: Średnia wysokość piku na liczbach pierwszych 
    vs średni szum na liczbach złożonych.
    """
    derivative = np.diff(pi_qm)
    primes_signal = []
    noise_signal = []
    
    for i in range(len(derivative)):
        val = x_range[0] + i + 1
        # Ignorujemy małe liczby, gdzie definicje są chwiejne
        if val < 2: continue
        
        if sympy.isprime(int(val)):
            primes_signal.append(derivative[i])
        elif not is_perfect_power(val):
            noise_signal.append(abs(derivative[i]))
            
    avg_signal = np.mean(primes_signal) if primes_signal else 0
    # Dodajemy epsilon, żeby nie dzielić przez zero w idealnych warunkach
    avg_noise = np.mean(noise_signal) if noise_signal else 1e-6
    
    return avg_signal / avg_noise

def run_heisenberg_test():
    print("=== QUANTUM LIMIT VERIFICATION TEST ===")
    
    # 1. Dane i Limit Teoretyczny
    all_zeros = load_data_full()
    if len(all_zeros) == 0:
        print("Error: No data found.")
        return

    gamma_max = all_zeros[-1]
    # LIMIT HEISENBERGA: X < gamma / 2pi
    heisenberg_limit = gamma_max / (2 * np.pi)
    
    print(f"--> Gamma Max: {gamma_max:.2f}")
    print(f"--> CALCULATED HEISENBERG LIMIT: X = {int(heisenberg_limit):,}")
    
    # 2. Skanowanie
    scanner = QuantumPrimeScanner(all_zeros)
    snr_results = []
    x_points = []
    
    print(f"\n{'Target X':<12} | {'SNR':<8} | {'Regime'}")
    print("-" * 40)
    
    for X in TARGETS:
        window = 200
        scan_range = np.arange(X, X + window)
        
        # Obliczenia
        pi_vals = scanner.compute_pi_qm(scan_range, desc=None)
        snr = measure_snr(pi_vals, scan_range)
        
        # Określenie reżimu (poniżej czy powyżej limitu)
        regime = "Sub-limit" if X < heisenberg_limit else "Post-limit (Decoherence)"
        if X > heisenberg_limit * 1.5: regime = "Noise"
        
        print(f"{X:<12} | {snr:<8.4f} | {regime}")
        
        snr_results.append(snr)
        x_points.append(X)

    # 3. Wizualizacja
    plt.figure(figsize=(12, 7))
    
    # Linia SNR
    plt.plot(x_points, snr_results, 'bo-', linewidth=2, label='Measured SNR')
    
    # Linia Limitu
    plt.axvline(heisenberg_limit, color='r', linestyle='--', linewidth=2, label=f'Heisenberg Limit (X={int(heisenberg_limit)})')
    
    # Obszary
    plt.axvspan(0, heisenberg_limit, color='green', alpha=0.1, label='Coherent Regime')
    plt.axvspan(heisenberg_limit, max(TARGETS), color='red', alpha=0.1, label='Decoherent Regime')
    
    plt.xscale('log') # Skala logarytmiczna, żeby widzieć 300 i 3mln na jednym wykresie
    plt.xlabel('Number Line Position (X)')
    plt.ylabel('Signal-to-Noise Ratio (SNR)')
    plt.title(f'Verification of Heisenberg-Riemann Limit (N={len(all_zeros):,} Zeros)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.4)
    
    output_filename = "Fig6_HeisenbergLimit_FullTest.png"
    plt.savefig(output_filename)
    print(f"\n--> Plot saved to {output_filename}")

if __name__ == "__main__":
    run_heisenberg_test()