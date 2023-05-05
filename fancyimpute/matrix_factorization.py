# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from sklearn.utils import check_array
from .solver import Solver
from .common import masked_mae


class MatrixFactorization(Solver):
    def __init__(
        self,
        rank=40,
        learning_rate=0.01,
        max_iters=50,
        shrinkage_value=0,
        verbose=True,
    ):
        """
        Train a matrix factorization model to predict empty
        entries in a matrix. Mostly copied (with permission) from:
        https://blog.insightdatascience.com/explicit-matrix-factorization-als-sgd-and-all-that-jazz-b00e4d9b21ea

        Params
        =====+
        rank : (int)
            Number of latent factors to use in matrix
            factorization model

        learning_rate : (float)
            Learning rate for optimizer

        max_iters : (int)
            Number of max_iters to train for

        shrinkage_value : (float)
            Regularization term for sgd penalty

        verbose : (bool)
            Whether or not to printout training progress
        """
        Solver.__init__(self)
        self.rank = rank
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.shrinkage_value = shrinkage_value
        self._v = verbose

    def solve(self, X, missing_mask):
        """ Train model for max_iters iterations from scratch."""
        X = check_array(X, force_all_finite=False)

        # shape data to fit into keras model
        (n_samples, n_features) = X.shape
        observed_mask = ~missing_mask
        training_indices = list(zip(*np.where(observed_mask)))

        self.user_vecs = np.random.normal(scale=1.0 / self.rank, size=(n_samples, self.rank))
        self.item_vecs = np.random.normal(scale=1.0 / self.rank, size=(n_features, self.rank))

        for i in range(self.max_iters):
            # to do: early stopping
            if (i + 1) % 10 == 0 and self._v:
                X_reconstruction = self.predict_all()
                mae = masked_mae(X_true=X, X_pred=X_reconstruction, mask=observed_mask)
                print("[MatrixFactorization] Iter %d: observed MAE=%0.6f rank=%d" % (i + 1, mae, self.rank))

            np.random.shuffle(training_indices)
            self.sgd(X, training_indices)
            i += 1

        X_filled = self.predict_all()
        return X_filled

    def sgd(self, X, training_indices):
        # to do: batch learning
        for (u, i) in training_indices:
            prediction = self.predict(u, i)
            e = X[u, i] - prediction  # error

            # Update latent factors
            self.user_vecs[u, :] += self.learning_rate * (
                e * self.item_vecs[i, :] - self.shrinkage_value * self.user_vecs[u, :]
            )
            self.item_vecs[i, :] += self.learning_rate * (
                e * self.user_vecs[u, :] - self.shrinkage_value * self.item_vecs[i, :]
            )

    def predict(self, u, i):
        """ Single user and item prediction."""
        prediction = self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
        return prediction

    def predict_all(self):
        """ Predict ratings for every user and item."""
        predictions = self.user_vecs.dot(self.item_vecs.T)
        return predictions
