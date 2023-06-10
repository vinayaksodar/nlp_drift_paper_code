import numpy as np
from scipy.stats import ks_2samp
from scipy.stats import entropy
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import svm
from sklearn.kernel_approximation import Nystroem
from scipy import stats


class DriftDetector:
    """
    A module to detect drift in embeddings using KS and MMD tests.
    """

    def __init__(self, reference_embeddings):
        """
        Initialize the DriftDetector.

        Args:
            reference_embeddings (numpy.ndarray): Reference embeddings for drift detection.
        """
        self.reference_embeddings = reference_embeddings

    import numpy as np
from scipy import stats


class DriftDetector:
    """
    A module to detect drift in embeddings using different drift detection methods.
    """

    def __init__(self):
        """
        Initialize the DriftDetector.
        """
        self.reference_embedding = None

    def set_reference_embedding(self, reference_embedding):
        """
        Set the reference embedding to compare against.

        Args:
            reference_embedding (numpy.ndarray): Reference embedding array of shape (num_samples, embedding_dim).

        Returns:
            None
        """
        self.reference_embedding = reference_embedding

    def ks_test(self, embeddings):
        """
        Perform the Kolmogorov-Smirnov (KS) test to detect drift in the embeddings.

        Args:
            embeddings (numpy.ndarray): Embeddings array of shape (num_samples, embedding_dim).

        Returns:
            overall_p_value (float): Overall p value of the drift
        """
        if self.reference_embedding is None:
            raise ValueError("No reference embedding set. Please set a reference embedding first.")

        ks_statistics = []
        p_values = []
        n_dims = embeddings.shape[1]

        for dim in range(n_dims):
            ks_stat, p_value = stats.ks_2samp(self.reference_embedding[:, dim], embeddings[:, dim])
            ks_statistics.append(ks_stat)
            p_values.append(p_value)

        # Apply Bonferroni correction
        alpha = 0.05  # Significance level
        n_tests = n_dims  # Number of hypothesis tests
        p_values_corrected = np.array(p_values) * n_tests
        p_values_corrected = np.minimum(p_values_corrected, 1.0)  # Ensure p-values do not exceed 1.0

        #We take min below to check if we can reject atleast one hypothesis with an errorrate < alpha, see Bonferroni wikipedia
        overall_p_value = np.min(p_values_corrected) 

        return overall_p_value


    def mmd_test(self, embeddings, kernel=None):
            """
            Perform the Maximum Mean Discrepancy (MMD) test to detect drift in the embeddings.

            Args:
                embeddings (numpy.ndarray): Embeddings array of shape (num_samples, embedding_dim).
                kernel (str or callable): Kernel to be used for MMD computation. If None, the 'rbf' kernel is used.

            Returns:
                float: p-value indicating the level of significance for drift detection.
            """
            if self.reference_embedding is None:
                raise ValueError("No reference embedding set. Please set a reference embedding first.")

            reference_distances = pairwise_distances(self.reference_embedding, metric='euclidean')
            current_distances = pairwise_distances(embeddings, self.reference_embedding, metric='euclidean')

            if kernel is None:
                kernel = 'rbf'

            mmd = self._compute_mmd(reference_distances, current_distances, kernel)

            # Perform two-sample permutation test to estimate p-value
            n_permutations = 1000
            permutations = np.random.permutation(embeddings)
            mmd_permuted = []
            for i in range(n_permutations):
                permuted_distances = pairwise_distances(permutations[i], self.reference_embedding, metric='euclidean')
                mmd_permuted.append(self._compute_mmd(reference_distances, permuted_distances, kernel))

            p_value = (np.sum(mmd_permuted > mmd) + 1) / (n_permutations + 1)

            return p_value

    def kl_divergence_test(self, embeddings):
        """
        Perform the KL divergence test to detect drift in the embeddings.

        Args:
            embeddings (numpy.ndarray): Embeddings array of shape (num_samples, embedding_dim).

        Returns:
            float: p-value indicating the level of significance for drift detection.
        """
        if self.reference_embedding is None:
            raise ValueError("No reference embedding set. Please set a reference embedding first.")

        reference_distribution = self._compute_embedding_distribution(self.reference_embedding)
        current_distribution = self._compute_embedding_distribution(embeddings)

        kl_divergence = entropy(reference_distribution, current_distribution)

        # Perform two-sample KS test to estimate p-value
        _, p_value = stats.ks_2samp(reference_distribution, current_distribution)

        return p_value

    def _compute_embedding_distribution(self, embeddings):
        """
        Compute the distribution of embeddings by aggregating the embeddings along the embedding dimension.

        Args:
            embeddings (numpy.ndarray): Embeddings array of shape (num_samples, embedding_dim).

        Returns:
            numpy.ndarray: Probability distribution of the embeddings.
        """
        embedding_sum = np.sum(embeddings, axis=1)
        embedding_sum /= np.sum(embedding_sum)

        return embedding_sum

    def _compute_mmd(self, X, Y, kernel):
        """
        Compute the Maximum Mean Discrepancy (MMD) between two sets of samples.

        Args:
            X (numpy.ndarray): Samples from the first distribution of shape (n_samples_X, n_features).
            Y (numpy.ndarray): Samples from the second distribution of shape (n_samples_Y, n_features).
            kernel (str or callable): Kernel to be used for MMD computation.

        Returns:
            float: Maximum Mean Discrepancy (MMD) value.
        """
        nystroem = Nystroem(kernel=kernel, n_components=min(X.shape[0], Y.shape[0]))
        X_kernel = nystroem.fit_transform(X)
        Y_kernel = nystroem.transform(Y)

        svm_classifier = svm.SVC(kernel='linear')
        svm_classifier.fit(X_kernel, np.ones(X_kernel.shape[0]))
        mmd = np.mean(svm_classifier.decision_function(Y_kernel))

        return mmd



"""
# Example usage

# Load the reference embeddings from a file
reference_embeddings = np.load('reference_embeddings.npy')

# Create an instance of DriftDetector
drift_detector = DriftDetector(reference_embeddings)

# Load the new embeddings from a file
new_embeddings = np.load('new_embeddings.npy')

# Perform KS test
ks_stat, p_value = drift_detector.detect_ks_test(new_embeddings)
print(f"KS Statistic: {ks_stat}")
print(f"KS p-value: {p_value}")

# Perform MMD test
mmd_stat = drift_detector.detect_mmd_test(new_embeddings)
print(f"MMD Statistic: {mmd_stat}")

"""