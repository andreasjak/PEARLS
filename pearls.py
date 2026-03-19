"""
PEARLS: Pitch Estimation using dictionary-Adaptive Recursive Least Squares

This is a Python implementation of the PEARLS algorithm described in 
"Online Estimation of Multiple Harmonic Signals" by Elvander et. al., 
published in IEEE/ACM Transactions on Audio, Language, and Speech Processing.
DOI: 10.1109/TASLP.2016.2634118
"""

import numpy as np
import scipy as scp
from typing import Tuple, Optional
import warnings


def pearls(d: np.ndarray, lambda_val: float, rls_xi: float, Lmax: int, 
           fs: float, fmin: float, fmax: float, fdist: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    PEARLS algorithm for online multi-pitch estimation.
    
    Parameters
    ----------
    d : np.ndarray
        Complex-valued signal (N x 1), typically the analytic signal
    lambda_val : float
        Forgetting factor on interval (0, 1), e.g., 0.995
    rls_xi : float
        Smoothness parameter (called xi in paper), e.g., 10000
    Lmax : int
        Maximum number of harmonics for each pitch, e.g., 10
    fs : float
        Sampling frequency in Hz
    fmin : float
        Minimum pitch frequency in Hz, e.g., 80
    fmax : float
        Maximum pitch frequency in Hz, e.g., 400
    fdist : float
        Initial pitch grid resolution in Hz, e.g., 5
        
    Returns
    -------
    w_rls_hist : np.ndarray
        Trajectory of filter coefficients (N x P*Lmax)
    fpgrid_hist : np.ndarray
        Trajectory of candidate pitch frequencies (N x P)
    """
    
    # ========== SETTINGS AND INITIALIZATION ==========
    
    # Dictionary learning settings
    do_dictionary_learning = True
    do_active_update = True
    
    # Number of samples for dictionary update (e.g., 45 ms)
    nbr_samples_for_pitch = int(np.floor(45e-3 * fs))
    
    # Speed-up settings
    waiting_period = 9e-3  # The waiting period during which a pitch block can be excluded from updating (s).
    block_update_threshold = int(np.floor(fs * waiting_period)) # As above, expressed in nbr of samples.
    zero_update_threshold = int(np.floor(block_update_threshold / 10))
    speed_up_horizon = block_update_threshold
    
    # Dictionary length stored in memory
    dictionary_length = 2000
    
    # Initial penalty parameters
    gamma = 4
    gamma2 = 80
    
    # Proximal gradient settings
    step_size = 1e-4
    max_iter = 20
    
    # Print progress
    do_print = True
    if do_print:
        print("This is pearls_copy.py")
    
    # ========== INITIALIZATION ==========
    
    N = len(d)
    fpgrid = np.arange(fmin, fmax + 1, fdist, dtype=float)
    P = len(fpgrid)
    nbr_of_variables = P * Lmax
    
    # Frequency matrix
    freq_mat = np.outer(np.arange(1, Lmax + 1), fpgrid)
    
    # Time vectors
    t = np.arange(N)
    t_temp = np.arange(dictionary_length)
    
    # Dictionary initialization
    A_inner = 2 * np.pi * np.outer(t_temp, freq_mat.ravel(order='F')) / fs
    # A will be used in a circular manner, where the first dictionary_length time steps
    # are the current while the last dictionary_length time steps are the previous. 
    # This allows for use of neative indexing if you need to access a time earlier than
    # dictionary time 0.
    A = np.zeros((2 * dictionary_length, P * Lmax), dtype=complex)
    A[:dictionary_length, :] = np.exp(1j * A_inner)
    A[dictionary_length:, :] = A[:dictionary_length, :]
    
    # Window for gamma update
    Delta = int(np.floor(np.log(0.01) / np.log(lambda_val)))
    
    # ========== INITIALIZE VARIABLES ==========
    
    # RLS initialization
    R = np.zeros((nbr_of_variables, nbr_of_variables), dtype=complex)
    r = np.zeros(nbr_of_variables, dtype=complex)
    w_hat = np.zeros(nbr_of_variables, dtype=complex)
    w_rls = np.zeros(nbr_of_variables, dtype=complex)
    
    # History storage
    w_rls_hist = np.zeros((nbr_of_variables, N), dtype=complex)
    fpgrid_hist = np.zeros((P, N), dtype=float)
    
    # Block update counters
    block_not_updated_since = np.zeros(P, dtype=int)
    has_been_untouched_since = np.zeros(P, dtype=int)
    
    # Active/inactive blocks
    active_blocks = np.arange(P)
    inactive_blocks = np.array([], dtype=int)
    
    # Active indices
    active_indices = np.arange(P * Lmax)
    index_matrix = np.arange(P * Lmax).reshape(Lmax, P, order='F')
    
    # ========== ALGORITHM ==========
    
    for n in range(N):
        if do_print and n % 100 == 0:
            print(f'\r{n} of {N} ', end='', flush=True)
            
        
        # Save current grid
        fpgrid_hist[:, n] = fpgrid
        
        # ========== NEW SAMPLE ==========
        sample_index = n % dictionary_length 
        xn = A[sample_index, :]
        
        # Update dictionary when cycle completes
        if (n+1) % dictionary_length == 0:
            A[dictionary_length:, :] = A[:dictionary_length, :]
            upper_time_index = min(N, n + dictionary_length)

            # If upper index wanted to exceed N we set the last part of t_temp to zero.

            t_temp = np.zeros_like(t_temp)
            t_temp[:upper_time_index - (n + 1)] = t[(n + 1):upper_time_index]
                
            A_inner = 2 * np.pi * np.outer(t_temp, freq_mat.ravel(order='F')) / fs
            A[:dictionary_length, :] = np.exp(1j * A_inner)
            
        dn = d[n]
        
        # RLS update
        R = lambda_val * R + np.outer(xn, xn.conj())
        r = lambda_val * r + xn * np.conj(dn)
        
        # ========== UPDATE PENALTY PARAMETERS ==========
        if n >= Delta and (n % 401) == 0:
            inner_prod_indices = np.arange(n - Delta + 1, n + 1)
            lambda_fact = lambda_val ** np.arange(Delta - 1, -1, -1)
            
            A_inner_prod_indices = np.arange(sample_index - Delta + 1, sample_index + 1)
            max_norm = np.max(np.abs(
                A[A_inner_prod_indices, :].conj().T @ (d[inner_prod_indices] * lambda_fact)
            ))
                
            gamma = 0.1 * max_norm
            gamma2 = 1.0 * max_norm
            
        # ========== ACTIVE SET UPDATE ==========
        if do_active_update:
            activation_candidates = np.where(block_not_updated_since > block_update_threshold)[0]
            
            if len(activation_candidates) > 0:
                active_blocks = np.sort(np.union1d(active_blocks, activation_candidates))
                inactive_blocks = np.setdiff1d(np.arange(P), active_blocks)
                new_indices = index_matrix[:, activation_candidates].ravel(order='F')
                active_indices = np.sort(np.union1d(active_indices, new_indices))
                
                block_not_updated_since[active_blocks] = 0
                
            block_not_updated_since[inactive_blocks] += 1
            
        # ========== PROXIMAL GRADIENT UPDATE ==========
        w_ell = w_hat[active_indices].copy()
        R_small = R[np.ix_(active_indices, active_indices)]
        r_small = r[active_indices]
        
        w_ell = proximal_gradient_update(
            w_ell, R_small, r_small, len(active_blocks), Lmax, 
            gamma, gamma2, max_iter, step_size
        )
        
        w_hat[active_indices] = w_ell
        
        # ========== RLS FILTER UPDATE ==========
        if n > 100:
            w_rls_new = rls_update(w_rls[active_indices], R_small, r_small, Lmax, rls_xi)
            w_rls = np.zeros(nbr_of_variables, dtype=complex)
            w_rls[active_indices] = w_rls_new
            
        w_rls_hist[:, n] = w_rls
        
        # ========== SET INACTIVE BLOCKS TO ZERO ==========
        if do_active_update:
            has_been_untouched_since += 1
            
            if n > speed_up_horizon:
                zero_candidates = np.where(has_been_untouched_since > zero_update_threshold)[0]
                
                if len(zero_candidates) > 0:
                    w_norms = np.linalg.norm(
                        w_hat.reshape(Lmax, P, order='F'), axis=0
                    )
                    set_to_zero = np.where(w_norms < 0.05)[0]
                    set_to_zero = np.intersect1d(set_to_zero, zero_candidates)
                    
                    if len(set_to_zero) > 0:
                        inactive_blocks = np.sort(np.union1d(set_to_zero, inactive_blocks))
                        active_blocks = np.setdiff1d(np.arange(P), inactive_blocks)
                        active_indices = index_matrix[:, active_blocks].ravel(order='F')
                        active_indices = np.sort(active_indices)
                        has_been_untouched_since[set_to_zero] = 0
                        
                has_been_untouched_since += 1
                
        # ========== DICTIONARY LEARNING ==========
        if do_dictionary_learning and n >= 1000 and ((n - 1) % 100) == 0:
            w_norms = np.linalg.norm(w_hat.reshape(Lmax, P, order='F'), axis=0)
            
            if np.max(w_norms) > 0.01:
                current_index_time = n
                update_horizon = 600
                pitch_limit = fdist / 2

                # Moving forward either the desired update horizon, the remaining samples in the signal, or the remaining samples in the dictionary
                forward_steps = min(update_horizon, N - 1 - current_index_time, dictionary_length - 1 - sample_index)
                stop_index_time = current_index_time + forward_steps
                stop_index_curr_A = sample_index + forward_steps

                # Moving backward either the desired number of samples for pitch estimation, the current time index, or the existing samples in the dictionary (allowing for wrap-around)
                backward_steps = min(nbr_samples_for_pitch, current_index_time, sample_index + dictionary_length - 1)
                start_index_time = current_index_time - backward_steps
                start_index_curr_A = sample_index - backward_steps

                ref_signal = d[start_index_time:current_index_time + 1]
  
                A, fpgrid, w_hat = \
                    dictionary_update(
                        w_rls, ref_signal, pitch_limit, A,
                        fpgrid, t, fs, Lmax, P, start_index_time,
                        stop_index_time, start_index_curr_A,
                        stop_index_curr_A
                    )
                        
    return w_rls_hist, fpgrid_hist


def proximal_gradient_update(w_in: np.ndarray, Rn: np.ndarray, rn: np.ndarray,
                             P: int, Lmax: int, gamma: float, gamma2: float,
                             max_iter: int, step_size: float) -> np.ndarray:
    """
    Proximal gradient update for sparse group LASSO.
    
    Parameters
    ----------
    w_in : np.ndarray
        Input filter coefficients
    Rn : np.ndarray
        Correlation matrix
    rn : np.ndarray
        Cross-correlation vector
    P : int
        Number of pitch blocks
    Lmax : int
        Maximum harmonics per pitch
    gamma : float
        L1 penalty parameter
    gamma2 : float
        Group penalty parameter
    max_iter : int
        Maximum iterations
    step_size : float
        Gradient step size
        
    Returns
    -------
    w_out : np.ndarray
        Updated filter coefficients
    """
    w_ell = w_in.copy()
    first_harm_amps = np.ones(P)
    
    for ell in range(max_iter):
        # Gradient step
        temp_gradient = -rn + Rn @ w_ell
        r_ell = w_ell - step_size * temp_gradient
        r_ell = soft_threshold(r_ell, gamma * step_size)
        
        # Projection for each block
        for k_block in range(P):
            temp_indices = np.arange(k_block * Lmax, (k_block + 1) * Lmax)
            temp_w = r_ell[temp_indices]
            
            gamma2_temp = gamma2 * max(1, min(1000, 1 / (np.abs(first_harm_amps[k_block]) + 1e-5)))
            temp_norm = np.linalg.norm(temp_w)
            temp = max(temp_norm - gamma2_temp * step_size**2, 0)
            
            w_ell[temp_indices] = temp / (temp + gamma2_temp * step_size**2) * temp_w
            
        first_harm_amps = w_ell[::Lmax]
        
    return w_ell


def soft_threshold(x: np.ndarray, gamma: float) -> np.ndarray:
    """
    Soft thresholding operator for L1 penalty.
    
    Parameters
    ----------
    x : np.ndarray
        Input array
    gamma : float
        Threshold value
        
    Returns
    -------
    z : np.ndarray
        Thresholded array
    """
    temp_z = np.maximum(np.abs(x) - gamma, 0)
    z = temp_z / (temp_z + gamma) * x
    return z


def rls_update(w_old: np.ndarray, R: np.ndarray, r: np.ndarray, 
               Lmax: int, rls_lambda: float) -> np.ndarray:
    """
    Smooth RLS update to counter bias from sparse penalty.
    
    Parameters
    ----------
    w_old : np.ndarray
        Previous filter coefficients
    R : np.ndarray
        Correlation matrix
    r : np.ndarray
        Cross-correlation vector
    Lmax : int
        Maximum harmonics per pitch
    rls_lambda : float
        Smoothing parameter (xi in paper)
        
    Returns
    -------
    w_rls : np.ndarray
        Updated RLS filter coefficients
    """
    P = len(w_old) // Lmax
    w_rls = w_old.copy()
    lambda_I = rls_lambda * np.eye(Lmax)
    
    for k_pitch in range(P):
        temp_indices = np.arange(k_pitch * Lmax, (k_pitch + 1) * Lmax)
        
        other_indices = np.setdiff1d(np.arange(P * Lmax), temp_indices)
            
        R_temp = R[temp_indices, :]
        R_q = R_temp[:, other_indices]
        R_p = R_temp[:, temp_indices]
        
        r_p = r[temp_indices]
        r_p = r_p - R_q @ w_rls[other_indices]
        
        R_tilde = R_p + lambda_I
        r_tilde = r_p + rls_lambda * w_rls[temp_indices]
        
        w_rls[temp_indices] = scp.linalg.solve(R_tilde, r_tilde)
        
    return w_rls


def dictionary_update(w_hat: np.ndarray, ref_signal: np.ndarray, pitch_limit: float,
                     A: np.ndarray,
                     fpgrid: np.ndarray, t: np.ndarray, fs: float, Lmax: int, P: int,
                     start_index_time: int, stop_index_time: int,
                     start_index_curr_A: int, stop_index_curr_A: int,
                     ) -> Tuple:
    """
    Update the pitch frequency grid adaptively.
    
    Parameters
    ----------
    w_hat : np.ndarray
        Current filter coefficients
    ref_signal : np.ndarray
        Reference signal segment
    pitch_limit : float
        Search interval half-width in Hz
    A : np.ndarray
        Current dictionary
    A_inner : np.ndarray
        Dictionary phases
    A_inner_no_phase : np.ndarray
        Dictionary without phase
    fpgrid : np.ndarray
        Current frequency grid
    t : np.ndarray
        Time vector
    fs : float
        Sampling frequency
    Lmax : int
        Max harmonics
    P : int
        Number of pitches
    dictionary_length : int
        Dictionary buffer length
    start_index_time : int
        Start time index
    stop_index_time : int
        Stop time index
    curr_index_curr_A : int
        Current dictionary index
    start_index_curr_A : int
        Start index in current A
    stop_index_curr_A : int
        Stop index in current A
    A_old : np.ndarray, optional
        Previous dictionary
    start_index_old_A : int, optional
        Start index in old A
        
    Returns
    -------
    Tuple of updated dictionary components and change flag
    """
    w_reshape = w_hat.reshape(Lmax, P, order='F')
    w_norms = np.linalg.norm(w_reshape, axis=0)
    
    A_new = A
    fpgrid_new = fpgrid.copy()
    
    t_temp = t[start_index_time:stop_index_time + 1]
    
    # Use only current dictionary
    change_indices_curr_A = np.arange(start_index_curr_A, stop_index_curr_A + 1)
    index_for_current_A = 0

        
    # Find biggest present pitches
    peak_indices, _ = find_peaks(w_norms)
    
    if len(peak_indices) == 0:
        return A_new, fpgrid_new, w_hat
        
    sorted_indices = np.argsort(w_norms[peak_indices])[::-1]
    temp_indices = peak_indices[sorted_indices]
    
    # Keep only big peaks
    temp_indices = temp_indices[w_norms[temp_indices] >= 0.05 * np.max(w_norms[temp_indices])]
    
    if len(temp_indices) == 0:
        return A_new, fpgrid_new, w_hat
        
    for biggest_f0_index in temp_indices:
        
        max_harm_amp = np.max(np.abs(w_reshape[:, biggest_f0_index]))
        nz_harmonics = np.where(np.abs(w_reshape[:, biggest_f0_index]) > 0.2 * max_harm_amp)[0]
        
        pitch_int = np.array([
            fpgrid_new[biggest_f0_index] - pitch_limit,
            fpgrid_new[biggest_f0_index] + pitch_limit
        ])
        
        new_grid_point = interval_search_anls(ref_signal, np.max(nz_harmonics) + 1, pitch_int, fs)
        
        if new_grid_point is None:
            continue
            
        fpgrid_new[biggest_f0_index] = new_grid_point
        
        # Get harmonic indices
        harmonic_indices = np.arange(biggest_f0_index * Lmax, (biggest_f0_index + 1) * Lmax)
        
        # Update dictionary
        A_inner_no_phase_temp = 2 * np.pi * np.outer(t_temp, np.arange(1, Lmax + 1)) * new_grid_point / fs
        A_inner_temp = phase_update(ref_signal, A_inner_no_phase_temp, 0, len(ref_signal) - 1,
                                    1, Lmax, Lmax)
        A_temp = np.exp(1j * A_inner_temp)
        
        # Update matrices
        temp_index_update = index_for_current_A + len(change_indices_curr_A)
            
        A_new[np.ix_(change_indices_curr_A, harmonic_indices)] = \
            A_temp[index_for_current_A:temp_index_update, :]
            
                
    return A_new, fpgrid_new, w_hat


def interval_search_anls(x: np.ndarray, L: int, f0_lim: np.ndarray, 
                        fs: float, nfft: int = 2**20) -> Optional[float]:
    """
    Approximate non-linear least squares for pitch frequency update.
    
    Parameters
    ----------
    x : np.ndarray
        Signal segment
    L : int
        Number of harmonics
    f0_lim : np.ndarray
        Frequency search interval [min, max] in Hz
    fs : float
        Sampling frequency
    nfft : int
        FFT size
        
    Returns
    -------
    f0 : float or None
        Updated fundamental frequency in Hz
    """
    freq_vec = np.arange(nfft) / nfft * fs
    N = len(x)
    
    a = np.where(freq_vec >= f0_lim[0])[0]
    b = np.where(freq_vec <= f0_lim[1])[0]
    
    if len(a) == 0 or len(b) == 0:
        return None
        
    a = a[0]
    b = b[-1]
    
    m = int(np.floor((a + b) / 2))
    lambda_val = m - 1
    mu = m + 1
    tol = 3
    
    while b - a > tol:
        F_lambda = _compute_F(x, N, nfft, lambda_val, L)
        F_mu = _compute_F(x, N, nfft, mu, L)
        
        if F_lambda > F_mu:
            b = mu
        else:
            a = lambda_val
            
        m = int(np.floor((a + b) / 2))
        lambda_val = m - 1
        mu = m + 1
        
    f0 = freq_vec[int(np.floor((a + b) / 2))]
    return f0


def _compute_F(x: np.ndarray, N: int, nfft: int, k: int, L: int) -> float:
    """Compute objective function for interval search."""
    val = 0.0
    n = np.arange(N)
    
    for k_harmonic in range(1, L + 1):
        basis = np.exp(-2j * np.pi * k * k_harmonic / nfft * n)
        val += np.abs(np.dot(basis, x))**2
        
    return val


def phase_update(ref_signal: np.ndarray, A_inner: np.ndarray, start_index: int,
                stop_index: int, update_indices: int, Lmax: int, 
                nbr_harmonics: int) -> np.ndarray:
    """
    Update phases in dictionary.
    
    Parameters
    ----------
    ref_signal : np.ndarray
        Reference signal
    A_inner : np.ndarray
        Dictionary phases
    start_index : int
        Start index
    stop_index : int
        Stop index
    update_indices : int
        Indices to update
    Lmax : int
        Max harmonics
    nbr_harmonics : int
        Number of harmonics to update
        
    Returns
    -------
    A_inner_out : np.ndarray
        Updated dictionary phases
    """
    A_inner_out = A_inner.copy()
    
    for k_pitch in range(update_indices):
        for kk in range(nbr_harmonics):
            temp_A_inner = A_inner[start_index:stop_index + 1, k_pitch * Lmax + kk]
            temp_A = np.exp(1j * temp_A_inner)
            temp_res = ref_signal / temp_A
            phi_est = np.angle(np.mean(temp_res))
            A_inner_out[start_index:, k_pitch * Lmax + kk] += phi_est
            
    return A_inner_out


def find_peaks(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple peak finding (local maxima).
    
    Parameters
    ----------
    x : np.ndarray
        Input signal
        
    Returns
    -------
    peak_indices : np.ndarray
        Indices of peaks
    peak_values : np.ndarray
        Values at peaks
    """
    peaks = []
    
    for i in range(1, len(x) - 1):
        if x[i] > x[i - 1] and x[i] > x[i + 1]:
            peaks.append(i)
            
    peak_indices = np.array(peaks, dtype=int)
    peak_values = x[peak_indices] if len(peak_indices) > 0 else np.array([])
    
    return peak_indices, peak_values
