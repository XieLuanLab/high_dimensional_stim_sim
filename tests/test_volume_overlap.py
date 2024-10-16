import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

from utils import compute_volume_overlap, plot_gaussian_ellipsoid


def test_compute_volume_overlap(plot_ellipsoids=False):
    # Test Case 1: Identical Gaussians
    mean_1 = np.array([0, 0, 0])
    cov_1 = np.eye(3)
    overlap, samples_1, samples_2 = compute_volume_overlap(mean_1, cov_1, mean_1, cov_1)
    if plot_ellipsoids:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        plot_every_n = 5
        plot_gaussian_ellipsoid(mean_1, cov_1, ax=ax)
        ax.scatter(
            samples_1[::plot_every_n, 0],
            samples_1[::plot_every_n, 1],
            samples_1[::plot_every_n, 2],
            c="C0",
            s=1,
        )
        ax.scatter(
            samples_2[::plot_every_n, 0],
            samples_2[::plot_every_n, 1],
            samples_2[::plot_every_n, 2],
            c="C1",
            s=1,
        )
        plt.title(f"Test 1 overlap: {overlap}")
    assert np.isclose(
        overlap, 1.0, atol=1e-5
    ), f"Expected overlap: ~1.0, Got: {overlap}"

    # Test Case 2: Disjoint Gaussians
    mean_2 = np.array([5, 5, 5])
    cov_2 = np.eye(3)
    overlap, samples_1, samples_2 = compute_volume_overlap(mean_1, cov_1, mean_2, cov_2)
    if plot_ellipsoids:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        plot_every_n = 5
        plot_gaussian_ellipsoid(mean_1, cov_1, ax=ax)
        plot_gaussian_ellipsoid(mean_2, cov_2, ax=ax)
        ax.scatter(
            samples_1[::plot_every_n, 0],
            samples_1[::plot_every_n, 1],
            samples_1[::plot_every_n, 2],
            c="C0",
            s=1,
        )
        ax.scatter(
            samples_2[::plot_every_n, 0],
            samples_2[::plot_every_n, 1],
            samples_2[::plot_every_n, 2],
            c="C1",
            s=1,
        )
        plt.title(f"Test 2 overlap: {overlap:.3f}")
    assert np.isclose(
        overlap, 0.0, atol=1e-5
    ), f"Expected overlap: ~0.0, Got: {overlap}"

    # Test Case 3: Gaussians with different mean and covariance but with overlap
    mean_3 = np.array([1.5, 2, 3])
    cov_3 = np.array([[2, 2, 1], [1, 2, 1], [1, 1, 4]])
    overlap, samples_1, samples_2 = compute_volume_overlap(mean_1, cov_1, mean_3, cov_3)
    if plot_ellipsoids:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        plot_every_n = 5
        plot_gaussian_ellipsoid(mean_1, cov_1, ax=ax)
        plot_gaussian_ellipsoid(mean_3, cov_3, ax=ax)
        ax.scatter(
            samples_1[::plot_every_n, 0],
            samples_1[::plot_every_n, 1],
            samples_1[::plot_every_n, 2],
            c="C0",
            s=1,
        )
        ax.scatter(
            samples_2[::plot_every_n, 0],
            samples_2[::plot_every_n, 1],
            samples_2[::plot_every_n, 2],
            c="C1",
            s=1,
        )
        plt.title(f"Test 3 overlap: {overlap:.3f}")
    assert (
        overlap > 0.0 and overlap < 1.0
    ), f"Expected overlap between 0 and 1, Got: {overlap}"

    print("All tests passed.")


# Run the test function
test_compute_volume_overlap(plot_ellipsoids=True)
