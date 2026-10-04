import numpy as np

def compute_t2i_metrics(errors_matrix, num_captions_per_image=5):
    """
    Text-to-Image retrieval evaluation (Image search from query caption).
    errors_matrix: (N_caps, N_imgs) where smaller values mean higher similarity / lower error.
    Returns: dict with R@1, R@5, R@10, and Mean Rank.
    """
    num_caps = errors_matrix.shape[0]
    ranks = np.zeros(num_caps)

    for i in range(num_caps):
        d_i = errors_matrix[i]
        # Sort image candidates by ascending error
        inds = np.argsort(d_i)
        gt_image_idx = i // num_captions_per_image

        # Find 1-based or 0-based rank of ground truth
        match_positions = np.where(inds == gt_image_idx)[0]
        if len(match_positions) > 0:
            rank = match_positions[0]
        else:
            rank = len(inds)
        ranks[i] = rank

    r1 = 100.0 * np.sum(ranks < 1) / num_caps
    r5 = 100.0 * np.sum(ranks < 5) / num_caps
    r10 = 100.0 * np.sum(ranks < 10) / num_caps
    mean_rank = float(np.mean(ranks) + 1.0) # 1-indexed mean rank

    return {
        'R@1': round(float(r1), 2),
        'R@5': round(float(r5), 2),
        'R@10': round(float(r10), 2),
        'Mean_r': round(mean_rank, 2)
    }


def compute_i2t_metrics(errors_matrix, num_captions_per_image=5):
    """
    Image-to-Text retrieval evaluation (Caption search from query image).
    errors_matrix: (N_caps, N_imgs)
    """
    num_imgs = errors_matrix.shape[1]
    ranks = np.zeros(num_imgs)

    for j in range(num_imgs):
        d_j = errors_matrix[:, j]
        inds = np.argsort(d_j)

        # Ground truth caption indices are j*5 to j*5 + 4
        gt_cap_indices = np.arange(j * num_captions_per_image, (j + 1) * num_captions_per_image)
        # Rank of the first correctly retrieved caption
        first_match = np.inf
        for pos, candidate in enumerate(inds):
            if candidate in gt_cap_indices:
                first_match = pos
                break
        ranks[j] = first_match

    r1 = 100.0 * np.sum(ranks < 1) / num_imgs
    r5 = 100.0 * np.sum(ranks < 5) / num_imgs
    r10 = 100.0 * np.sum(ranks < 10) / num_imgs
    mean_rank = float(np.mean(ranks) + 1.0)

    return {
        'R@1': round(float(r1), 2),
        'R@5': round(float(r5), 2),
        'R@10': round(float(r10), 2),
        'Mean_r': round(mean_rank, 2)
    }
