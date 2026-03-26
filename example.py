"""
Example usage of PEARLS algorithm - equivalent to mail.m
"""

import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from pearls import pearls
import cProfile

def main():

    profile = True
    """Run PEARLS on audio file."""
    
    # Read audio file
    # Note: Update path to your audio file
    audio_file = 'BachData/Bach/01-AchGottundHerr.wav'
    
    try:
        if profile:
            raise FileNotFoundError("Profiling mode - skipping audio file loading")
        fs, x = wavfile.read(audio_file)
        
        # Convert to float
        if x.dtype == np.int16:
            x = x.astype(np.float32) / 32768.0
        elif x.dtype == np.int32:
            x = x.astype(np.float32) / 2147483648.0
            
        # Convert stereo to mono if needed
        if len(x.shape) > 1:
            x = np.mean(x, axis=1)
            
    except FileNotFoundError:
        print(f"Audio file '{audio_file}' not found. Using synthetic signal instead.")
        # Create synthetic signal for demonstration
        fs = 44100
        duration = 0.5 
        t = np.linspace(0, duration, int(fs * duration))
        
        # Two pitches with harmonics
        f1, f2 = 220, 330  # A3 and E4
        # Added vibrato effect by modulating the frequency with a low-frequency sine wave
        x = (np.sin(2 * np.pi * (f1 + 2 * np.sin(2 * np.pi * 4 * t)) * t) + 
             0.5 * np.sin(2 * np.pi * 2 * (f1 + 2 * np.sin(2 * np.pi * 4 * t))* t) +
             0.5 * np.sin(2 * np.pi * f2 * t) +
             0.3 * np.sin(2 * np.pi * 2 * f2 * t))
        # x += 0.01 * np.random.randn(len(x))  # Add noise
        
    # Decimate by factor 4 (with anti-aliasing filter)
    y = signal.decimate(x, 4, ftype='iir')
    fs_new = fs / 4
    
    # Create analytic signal using Hilbert transform
    z = signal.hilbert(y)

    # z = z[:10_000]
    
    print(f"Processing signal: {len(z)} samples at {fs_new} Hz")
    print(f"Duration: {len(z) / fs_new:.2f} seconds")

    # Run PEARLS algorithm
    print("\nRunning PEARLS algorithm...")
    if profile:
        cProfile.runctx('pearls(d=z, lambda_val=0.995, rls_xi=10000, Lmax=10, fs=fs_new, fmin=80, fmax=400, fdist=5)',
                        globals(), locals(), "stats_improved")
        return

    w_rls_hist, fpgrid_hist, active_block_hist = pearls(
        d=z,
        lambda_val=0.995,
        rls_xi=10000,
        Lmax=10,
        fs=fs_new,
        fmin=40,
        fmax=800,
        fdist=10
    )
    
    print("PEARLS algorithm completed!")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Filter coefficients
    im1 = axes[0].imshow(np.abs(w_rls_hist), aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Filter Coefficients (w_rls_hist)')
    axes[0].set_xlabel('Time (samples)')
    axes[0].set_ylabel('Coefficient Index')
    plt.colorbar(im1, ax=axes[0], label='Magnitude')
    
    # Plot 2: Pitch frequency grid
    time_axis = np.arange(fpgrid_hist.shape[1]) / fs_new
    pitch_axis = np.arange(fpgrid_hist.shape[0])
    
    im2 = axes[1].imshow(fpgrid_hist, aspect='auto', origin='lower', 
                         extent=[time_axis[0], time_axis[-1], 0, fpgrid_hist.shape[0]],
                         cmap='viridis')
    axes[1].set_title('Pitch Frequency Grid (fpgrid_hist)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Pitch Index')
    plt.colorbar(im2, ax=axes[1], label='Frequency (Hz)')
    
    plt.tight_layout()
    plt.show()
    
    # Create pitch trajectory plot
    fig2, ax = plt.subplots(figsize=(12, 6))
    
    # Compute pitch energies
    Lmax = 10
    P = fpgrid_hist.shape[0]
    w_reshape = w_rls_hist.reshape(Lmax, P, -1, order='F')
    pitch_energies = np.linalg.norm(w_reshape, axis=0)
    
    # Plot active pitches
    threshold = 0.1 * np.max(pitch_energies)
    
    for p in range(P):
        active = pitch_energies[p, :] > threshold
        if np.any(active):
            t_active = time_axis[active]
            f_active = fpgrid_hist[p, active]
            scatter = ax.scatter(t_active, f_active, c=pitch_energies[p, active], 
                               s=20, alpha=0.6, cmap='plasma')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Active Pitch Trajectories')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Pitch Energy')
    
    plt.tight_layout()
    plt.show()

    plt.imshow(active_block_hist, aspect='auto', origin='lower', cmap='gray_r')
    plt.show()
    
    # # Save numerical results
    # np.savez('/mnt/user-data/outputs/pearls_output.npz',
    #          w_rls_hist=w_rls_hist,
    #          fpgrid_hist=fpgrid_hist,
    #          fs=fs_new,
    #          time=time_axis)
    # print("Numerical results saved to pearls_output.npz")
    
    return w_rls_hist, fpgrid_hist


if __name__ == '__main__':
    main()