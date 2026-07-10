import numpy as np
from sklearn.metrics import adjusted_rand_score

from ark.cluster.idec import fit_idec


def test_fit_idec_recovers_well_separated_clusters():
    rng = np.random.default_rng(0)
    # three well-separated blobs: a real, easy structure IDEC should recover
    # cleanly, the same sanity check its own pretrain-then-joint-train loop
    # needs to pass before trusting it on harder, real load-shape data.
    blob_a = rng.normal(loc=-8, scale=0.3, size=(15, 4))
    blob_b = rng.normal(loc=0, scale=0.3, size=(15, 4))
    blob_c = rng.normal(loc=8, scale=0.3, size=(15, 4))
    X = np.vstack([blob_a, blob_b, blob_c]).astype(np.float32)
    true_labels = np.array([0] * 15 + [1] * 15 + [2] * 15)

    _, labels = fit_idec(X, n_clusters=3, pretrain_epochs=50, train_epochs=20, random_state=0)

    assert labels.shape == (45,)
    assert adjusted_rand_score(true_labels, labels) > 0.9
