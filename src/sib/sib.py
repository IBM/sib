# © Copyright IBM Corporation 2020.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import copy
import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.preprocessing import normalize
from sklearn.utils import check_random_state
from joblib import Parallel, delayed, effective_n_jobs
from .p_sib_optimizer import PSIBOptimizer
from .c_sib_optimizer import CSIBOptimizer

from time import time


class SIB(BaseEstimator, ClusterMixin, TransformerMixin):
    """sequential Information Bottleneck (sIB) clustering.

    Parameters
    ----------

    n_clusters : int
        The number of clusters to form as well as the number of
        centroids to generate.

    n_init : int, default=10
        Number of times the sIB algorithm will be run with different
        centroid seeds. The final result will be the initialization
        with highest mutual information between the clustering
        analysis and the vocabulary.

    max_iter : int, default=15
        Maximum number of iterations of the sIB algorithm for a
        single run.

    tol : float, default=0.02
        Relative tolerance with regards to number of centroid updates
        to declare convergence.

    verbose : int, default=0
        Verbosity mode.

    random_state : int, RandomState instance, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.

    n_jobs : int, default=None
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.

        ``-1`` means using all processors.

    uniform_prior : bool, default=True
        Determines whether all input vectors are assumed to have the
        same probability.

    inv_beta : double, default=0
        Currently undocumented.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.

    labels_ : ndarray of shape (n_samples)
        Labels of each point

    score_ : float
        Mutual information between the cluster analysis and the vocabulary.

    inertia_ : float
        The score value negated

    n_iter_ : int
        Number of iterations ran

    costs_ :  ndarray of shape (n_samples, n_clusters)
        The input samples transformed to o cluster-distance space

    """

    def __init__(self, n_clusters, random_state=None, n_jobs=1,
                 n_init=10, max_iter=15, tol=0.02, verbose=False,
                 inv_beta=0, uniform_prior=True, optimizer_type='C'):
        self.n_clusters = n_clusters
        self.uniform_prior = uniform_prior
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.inv_beta = inv_beta
        self.optimizer_type = optimizer_type

        self.xy = None
        self.xy_sum = None
        self.x_sum = None
        self.y_sum = None
        self.xy_log_sum = None

        self.ixy = None
        self.hy = None
        self.hx = None

        self.n_samples = -1
        self.n_features = -1

        self.partition_ = None
        self.score_ = None
        self.inertia_ = None
        self.n_iter_ = None
        self.cluster_centers_ = None
        self.labels_ = None
        self.costs_ = None

    def __str__(self):
        param_values = [("n_cluseters", self.n_clusters), ("n_jobs", self.n_jobs),
                        ("n_init", self.n_init), ("max_iter", self.max_iter),
                        ("tol", self.tol), ("random_state", self.random_state),
                        ("uniform_prior", self.uniform_prior), ("inv_beta", self.inv_beta),
                        ("optimizer_type", self.optimizer_type),
                        ("verbose", self.verbose)]

        return "sIB(" + ", ".join(name + "=" + str(value)
                                  for name, value in param_values) + ")"

    def fit(self, x):
        """Compute sIB clustering.

        Parameters
        ----------
        x : sparse matrix, shape=(n_samples, n_features)
            It is recommended to provide count vectors (un-normalized)

        Returns
        -------
        self
            Fitted estimator.
        """
        self.n_samples, self.n_features = x.shape

        if not self.n_samples > 1:
            raise ValueError("n_samples=%d should be > 1" % self.n_samples)

        if self.n_samples < self.n_clusters:
            raise ValueError("n_samples=%d should be >= n_clusters=%d"
                             % (self.n_samples, self.n_clusters))

        if x.min() < 0:
            raise ValueError("X's values should be >= 0")

        # prepare the values matrix and sum arrays
        self.xy, self.xy_sum, self.xy_log_sum, self.x_sum, self.y_sum = self.prepare_data(x)

        # calc the mutual info between x and y as well as the entropy of x and y
        self.ixy, self.hx, self.hy = self.calc_mi_entropy(self.xy, self.xy_sum, self.x_sum,
                                                          self.y_sum, self.xy_log_sum)

        random_state = check_random_state(self.random_state)

        if self.verbose:
            print("Initialization complete")

        # Main (restarts) loop
        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
        if effective_n_jobs(self.n_jobs) == 1 or self.n_init == 1:
            # For a single thread, less memory is needed if we just store one set
            # of the best results (as opposed to one set per run per thread).
            best_partition = None
            for i, seed in enumerate(seeds):
                # run sib once
                tmp_partition = self.sib_single(seed, run_id=(i if self.n_init > 1 else None))
                if best_partition is None or tmp_partition.score > best_partition.score:
                    best_partition = tmp_partition
        else:
            # parallelization of sib runs
            results = Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(self.sib_single)(random_state=seed, job_id=job_id)
                for job_id, seed in enumerate(seeds))
            scores = np.fromiter((T.score for T in results), float, self.n_init)
            best_partition = results[np.argmax(scores)]

        if self.verbose:
            ity_div_ixy = best_partition.ity / self.ixy
            ht_div_hx = best_partition.ht / self.hx
            print("sIB information stats on best partition:\n\tI(T;Y) = %.4f, H(T) = %.4f\n\t"
                  "I(T;Y)/I(X;Y) = %.4f\n\tH(T)/H(X) = %.4f" %
                  (best_partition.ity, best_partition.ht, ity_div_ixy, ht_div_hx))

        # Last updates
        self.partition_ = best_partition
        self.score_ = best_partition.score
        self.inertia_ = -self.score_
        self.n_iter_ = best_partition.n_iter
        self.cluster_centers_ = best_partition.t_centroid / best_partition.t_sum[:, None]
        self.labels_, self.costs_, _ = self.infer_labels_costs_score(
            self.n_samples, self.xy, self.xy_sum, self.x_sum)
        return self

    def prepare_data(self, x):
        if self.uniform_prior:
            # we normalize by l1 if we are asked to assign each sample with the same probability
            xy = normalize(x, norm='l1', axis=1, copy=True, return_norm=False)
        else:
            # otherwise, we will use the data as-is
            xy = x
        x_sum = xy.sum(axis=1)
        y_sum = xy.sum(axis=0)
        if issparse(x):
            x_sum = x_sum.A.ravel()
            y_sum = y_sum.A.ravel()
        xy_sum = x_sum.sum()
        xy_log_sum = np.log2(xy_sum)
        return xy, xy_sum, xy_log_sum, x_sum, y_sum

    def sib_single(self, random_state, job_id=None, run_id=None):
        # initialization: random generator, partition and optimizers
        random_state = check_random_state(random_state)
        partition = Partition(self.n_samples, self.n_features, self.n_clusters,
                              self.xy, self.x_sum, self.xy_sum, self.xy_log_sum,
                              self.hy, random_state)
        optimizer, v_optimizer = self.create_optimizers()

        # main loop of optimizing the partition
        self.report_status(partition, job_id, run_id)
        while not self.converged(partition):
            self.optimize(partition, optimizer, v_optimizer)
            self.report_status(partition, job_id, run_id)
            # partition.dump()

        self.report_convergence(partition, job_id, run_id)

        # final calculations
        partition.score = partition.ity - self.inv_beta * partition.ht

        # return the partition
        return partition

    def create_c_optimizer(self):
        return CSIBOptimizer(self.n_clusters, self.n_features,
                             self.n_samples, self.xy,
                             self.xy_sum, self.x_sum)

    def create_p_optimizer(self):
        return PSIBOptimizer(self.n_clusters, self.n_features,
                             self.n_samples, self.xy,
                             self.xy_sum, self.x_sum)

    def create_optimizers(self):
        if self.optimizer_type == 'C':
            optimizer = self.create_c_optimizer()
            v_optimizer = None
        elif self.optimizer_type == 'P':
            optimizer = self.create_p_optimizer()
            v_optimizer = None
        else:
            optimizer = self.create_c_optimizer()
            v_optimizer = self.create_p_optimizer()
        return optimizer, v_optimizer

    def report_status(self, partition, job_id, run_id):
        if self.verbose:
            print((("Job %2d, " % job_id) if job_id is not None else "") +
                  (("Run %2d, " % run_id) if run_id is not None else "") +
                  ("Iteration %2d, I(T;Y)=%.4f, H(T)=%.4f" %
                   (partition.n_iter, partition.ity, partition.ht)) +
                  ((", Updates=%.2f%%" % (partition.change_ratio * 100))
                   if partition.n_iter > 0 else ""))

    def report_convergence(self, partition, job_id, run_id):
        if self.verbose:
            print((("Job %2d, " % job_id) if job_id is not None else "") +
                  (("Run %2d, " % run_id) if run_id is not None else "") +
                  partition.convergence_str)

    def optimize(self, partition, optimizer, v_optimizer):
        x_permutation = partition.random_state.permutation(self.n_samples).astype(np.int32)
        # x_permutation = np.arange(self.n_samples)

        v_partition = None
        if v_optimizer:
            v_partition = copy.deepcopy(partition)

        partition.change_ratio, partition.ity, partition.ht = optimizer.optimize(
            x_permutation, partition.t_size, partition.t_sum, partition.t_log_sum,
            partition.t_centroid, partition.labels, partition.ity)

        if v_optimizer:
            v_partition.change_ratio, v_partition.ity, v_partition.ht = v_optimizer.optimize(
                x_permutation, v_partition.t_size, v_partition.t_sum, v_partition.t_log_sum,
                v_partition.t_centroid, v_partition.labels, v_partition.ity)
            assert np.allclose(partition.labels, v_partition.labels)
            assert np.allclose(partition.change_ratio, v_partition.change_ratio)
            assert np.allclose(partition.t_sum, v_partition.t_sum)
            assert np.allclose(partition.t_log_sum, v_partition.t_log_sum)
            assert np.allclose(partition.t_centroid, v_partition.t_centroid)
            assert np.allclose(partition.t_size, v_partition.t_size)
            assert np.allclose(partition.ity, v_partition.ity)
            assert np.allclose(partition.ht, v_partition.ht)

        partition.n_iter += 1
        if v_optimizer:
            v_partition.n_iter += 1

    def converged(self, partition):
        if partition.n_iter > 0 and partition.change_ratio <= self.tol:
            partition.convergence_str = "sIB converged in iteration %d with change=%.2f%%" \
                                        % (partition.n_iter, 100 * partition.change_ratio)
            return True
        elif partition.n_iter >= self.max_iter:
            partition.convergence_str = "sIB did NOT converge (change=%.2f%%), stopped due to max_iter=%d" \
                                        % (100 * partition.change_ratio, self.max_iter)
            return True
        else:
            return False

    @staticmethod
    def calc_mi_entropy(xy, xy_sum, x_sum, y_sum, xy_log_sum):
        hx = -np.dot(x_sum, np.log2(x_sum) - xy_log_sum) / xy_sum
        hy = -np.dot(y_sum, np.log2(y_sum) - xy_log_sum) / xy_sum
        xy = xy.data if issparse(xy) else xy[np.nonzero(xy)]
        hxy = -np.dot(xy, np.log2(xy) - xy_log_sum) / xy_sum
        return hx + hy - hxy, hx, hy

    def infer_labels_costs_score(self, n_samples, xy, xy_sum, x_sum):
        optimizer, v_optimizer = self.create_optimizers()
        labels = np.empty(n_samples, dtype=np.int32)
        costs = np.empty((n_samples, self.n_clusters))
        score = optimizer.infer(n_samples, xy, xy_sum, x_sum,
                                self.partition_.t_size,
                                self.partition_.t_sum,
                                self.partition_.t_log_sum,
                                self.partition_.t_centroid,
                                labels, costs)
        if v_optimizer:
            v_labels = np.empty(n_samples, dtype=np.int32)
            v_costs = np.empty((n_samples, self.n_clusters))
            v_score = v_optimizer.infer(n_samples, xy, xy_sum, x_sum,
                                        self.partition_.t_size,
                                        self.partition_.t_sum,
                                        self.partition_.t_log_sum,
                                        self.partition_.t_centroid,
                                        v_labels, v_costs)
            assert np.isclose(score, v_score)
            assert np.allclose(costs, v_costs)
            assert np.allclose(labels, v_labels)
        return labels, costs, score

    def fit_new_data(self, x):
        n_samples, _ = x.shape

        if not self.partition_:
            raise ValueError("Estimator SIB must be fitted before being used")

        if not self.n_samples > 1:
            raise ValueError("n_samples=%d should be > 1" % self.n_samples)

        # prepare the values matrix and sum arrays
        xy, xy_sum, _, x_sum, _ = self.prepare_data(x)

        return self.infer_labels_costs_score(n_samples, xy, xy_sum, x_sum)

    def fit_transform(self, x, y=None, sample_weight=None):
        """Compute clustering and transform x to cluster-distance space.

        Equivalent to fit(x).transform(x) but more efficient.

        Parameters
        ----------
        x : sparse matrix of shape (n_samples, n_features)
            New data to transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        X_new : array, shape [n_samples, n_clusters]
            X transformed in the new space.
        """
        self.fit(x)
        return self.costs_

    def fit_predict(self, x, y=None, sample_weight=None):
        """Compute cluster centers and predict cluster index for each sample.

        Equivalent to fit(x).predict(x) but more efficient.

        Parameters
        ----------
        x : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        self.fit(x)
        return self.labels_

    def transform(self, x):
        """Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster
        centers.  The array returned is always dense.

        Parameters
        ----------
        x : sparse matrix of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_new : array, shape [n_samples, k]
            X transformed in the new space.
        """
        labels, costs, score = self.fit_new_data(x)
        return costs

    def predict(self, x):
        """Predict the closest cluster each sample in x belongs to.

        Parameters
        ----------
        x : sparse matrix of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        labels, costs, score = self.fit_new_data(x)
        return labels

    def score(self, x):
        """The value of x on the algorithm objective. This is the sum
        of distances between each sample in x and the centroid of the
        cluster predicted for it.

        Parameters
        ----------
        x : sparse matrix of shape (n_samples, n_features)
            New data.

        Returns
        -------
        score : float
            The value of x on the algorithm objective.
        """

        labels, costs, score = self.fit_new_data(x)
        return score


class Partition:
    def __init__(self, n_samples, n_features, n_clusters, xy, x_sum, xy_sum, xy_log_sum, hy, random_state):
        # Produce a random partition as an initialization point
        self.labels = random_state.permutation(np.linspace(0, n_clusters, n_samples,
                                                           endpoint=False).astype(np.int32))

        # initialize the data structures based on the labels and the joint distribution
        self.t_size = np.zeros(n_clusters, dtype=np.int32)
        self.t_sum = np.zeros(n_clusters, dtype=x_sum.dtype)
        self.t_centroid = np.zeros((n_clusters, n_features), dtype=xy.dtype)
        sparse = issparse(xy)
        for i in range(n_samples):
            t = self.labels[i]
            v = xy[i, :]
            self.t_size[t] += 1
            self.t_sum[t] += x_sum[i]
            if sparse:
                self.t_centroid[t, v.indices] += v.data
            else:
                self.t_centroid[t, :] += v
        self.t_log_sum = np.log2(self.t_sum)

        # calculate information
        t_centroid = self.t_centroid[np.nonzero(self.t_centroid)]
        self.ht = -np.dot(self.t_sum, self.t_log_sum - xy_log_sum) / xy_sum
        self.hty = -np.dot(t_centroid, np.log2(t_centroid) - xy_log_sum) / xy_sum
        self.ity = self.ht + hy - self.hty

        # more initializations
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_features = n_features
        self.n_iter = 0
        self.change_ratio = 0
        self.score = None
        self.convergence_str = None

    def __str__(self):
        return " size: %d\n counter: %d\n convergence_str: %s" % (
            self.n_clusters, self.n_iter, self.convergence_str)
