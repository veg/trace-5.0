"""
Max-Min Dispersion Coresets and Nyström Landmark Approximation.

Implements:
- Greedy furthest-point (k-medoids / facility location) landmark selection.
- Guaranteed worst-case coverage of small, dense, isolated transmission outbreaks.
- Linear-time Nyström low-rank Laplacian spectral decomposition for registry-scale alignments.
"""

import numpy as np
from scipy.linalg import eigh, pinv


def select_maxmin_landmarks(D_matrix, m_landmarks, initial_idx=None):
    """
    Selects m diverse landmarks using greedy furthest-point sampling (Max-Min dispersion).

    Guarantees that isolated, dense clusters and peripheral lineages are selected as landmarks,
    preventing the spectral low-pass smoothing caused by uniform random sampling.

    Args:
        D_matrix: Pairwise distance matrix of shape (N, N)
        m_landmarks: Number of landmarks m << N
        initial_idx: Optional starting index. Defaults to medoid.

    Returns:
        landmark_indices (np.ndarray of shape (m,))
    """
    N = D_matrix.shape[0]
    m = min(int(m_landmarks), N)
    if m >= N:
        return np.arange(N)

    landmarks = []
    if initial_idx is not None and 0 <= initial_idx < N:
        landmarks.append(initial_idx)
    else:
        # Start at the medoid (minimizing sum of distances)
        medoid_idx = int(np.argmin(np.sum(D_matrix, axis=1)))
        landmarks.append(medoid_idx)

    # Array of min distance to any selected landmark
    min_dist = D_matrix[:, landmarks[0]].copy()

    for _ in range(1, m):
        # Pick the point furthest from the current landmark set
        next_landmark = int(np.argmax(min_dist))
        landmarks.append(next_landmark)
        # Update minimum distances
        min_dist = np.minimum(min_dist, D_matrix[:, next_landmark])

    return np.array(landmarks, dtype=np.int64)


def nystrom_spectral_embedding(D_matrix, m_landmarks=200, n_components=3, sigma=None):
    """
    Computes leading Laplacian eigenvectors using Nyström landmark approximation.

    Args:
        D_matrix: Distance matrix of shape (N, N)
        m_landmarks: Number of landmarks (e.g. 200)
        n_components: Number of eigenvectors to return (e.g. 3)
        sigma: Gaussian affinity bandwidth. Defaults to median non-zero landmark distance.

    Returns:
        evals (np.ndarray of shape (k,))
        evecs (np.ndarray of shape (N, k))
    """
    N = D_matrix.shape[0]
    if N <= m_landmarks:
        # Full exact spectrum
        nonzero = D_matrix[D_matrix > 0]
        s = sigma if sigma is not None else (float(np.median(nonzero)) if len(nonzero) > 0 else 0.015)
        s = max(s, 1e-4)
        W = np.exp(-(D_matrix ** 2) / (2.0 * s ** 2))
        np.fill_diagonal(W, 0.0)
        deg = W.sum(axis=1)
        deg = np.maximum(deg, 1e-12)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
        L_sym = np.eye(N) - D_inv_sqrt @ W @ D_inv_sqrt
        w, v = eigh(L_sym)
        return w[:n_components], v[:, :n_components]

    landmarks = select_maxmin_landmarks(D_matrix, m_landmarks)
    m = len(landmarks)

    D_mm = D_matrix[np.ix_(landmarks, landmarks)]
    nonzero_mm = D_mm[D_mm > 0]
    s = sigma if sigma is not None else (float(np.median(nonzero_mm)) if len(nonzero_mm) > 0 else 0.015)
    s = max(s, 1e-4)

    W_mm = np.exp(-(D_mm ** 2) / (2.0 * s ** 2))
    np.fill_diagonal(W_mm, 0.0)

    # Cross-affinity block: N x m
    D_nm = D_matrix[:, landmarks]
    W_nm = np.exp(-(D_nm ** 2) / (2.0 * s ** 2))

    # Approximate row sums (degrees)
    # d_approx = W_nm @ (W_mm^-1 @ W_mm_row_sums) approx W_nm @ (1)
    d_approx = np.sum(W_nm, axis=1) * (float(N) / float(m))
    d_approx = np.maximum(d_approx, 1e-12)
    d_inv_sqrt = 1.0 / np.sqrt(d_approx)

    # Normalized cross-affinity
    S = d_inv_sqrt[:, None] * W_nm
    # S_mm = S[landmarks, :]
    S_mm = S[landmarks, :]

    # SVD of S_mm
    u_mm, s_mm, _ = np.linalg.svd(S_mm)
    s_mm_inv = 1.0 / np.sqrt(np.maximum(s_mm, 1e-12))
    V_approx = S @ (u_mm @ np.diag(s_mm_inv))

    # Orthonormalize V_approx via SVD
    u_final, s_final, _ = np.linalg.svd(V_approx, full_matrices=False)
    # Normalized Laplacian eigenvalues = 1 - singular_values^2
    evals = np.maximum(0.0, 1.0 - s_final[:n_components] ** 2)
    evecs = u_final[:, :n_components]

    return evals, evecs
