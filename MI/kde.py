import numpy as np

def get_unique_probs(x):
    uniqueids = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    _, unique_inverse, unique_counts = np.unique(uniqueids, return_index=False, return_inverse=True, return_counts=True)
    return np.asarray(unique_counts / float(sum(unique_counts))), unique_inverse

def Kget_dists(X):
    """Keras code to compute the pairwise distance matrix for a set of
    vectors specifie by the matrix X.
    """
    x2 = np.sum(np.square(X), axis=1, keepdims=True)
    dists = x2 + x2.T - 2 * np.matmul(X, X.T)
    return dists

def entropy_estimator_kl(x, var):
    # KL-based upper bound on entropy of mixture of Gaussians with covariance matrix var * I
    #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    #  and Kolchinsky and Tracey, Nonlinear Information Bottleneck, 2017. Eq. 10
    dims, N = float(x.shape[1]), float(x.shape[0])
    dists = Kget_dists(x)
    dists2 = dists / (2*var)
    normconst = (dims / 2.0) * np.log(2 * np.pi * var)
    lprobs = np.log(np.sum(np.exp(-dists2), axis=1)) - np.log(N) - normconst
    h = -np.mean(lprobs)

    return dims/2 + h

def entropy_estimator_bd(x, var):
    # Bhattacharyya-based lower bound on entropy of mixture of Gaussians with covariance matrix var * I
    #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    dims, N = float(x.shape[1]), float(x.shape[0])
    val = entropy_estimator_kl(x,4*var)
    return val + np.log(0.25)*dims/2

def kde_condentropy(x, var):
    # Return entropy of a multivariate Gaussian, in nats
    dims = x.shape[1]
    return (dims/2.0)*(np.log(2*np.pi*var) + 1)


def mi_kde(h, inputdata, var=0.1):
    # function: compute the mutual information between the input and the final representation
    # h: hidden representation at the final layer
    # inputdata: the input attribute matrix X
    # var: noise variance used in estimate the mutual information in KDE

    nats2bits = float(1.0 / np.log(2))
    h_norm = np.sum(np.square(h), axis=1, keepdims=True)
    h_norm[h_norm == 0.] = 1e-3
    h = h / np.sqrt(h_norm)
    input_norm = np.sum(np.square(inputdata), axis=1, keepdims=True)
    input_norm[input_norm == 0.] = 1e-3
    inputdata = inputdata / np.sqrt(input_norm)

    # the entropy of the input
    entropy_input = entropy_estimator_bd(inputdata, var)

    # compute the entropy of input given the hidden representation at the final layer
    entropy_input_h = 0.
    indices = np.argmax(h, axis=1)
    indices = np.expand_dims(indices, axis=1)
    p_h, unique_inverse_h = get_unique_probs(indices)
    p_h = np.asarray(p_h).T
    for i in range(len(p_h)):
        labelixs = unique_inverse_h==i
        entropy_input_h += p_h[i] * entropy_estimator_bd(inputdata[labelixs, :], var)

    # the mutual information between the input and the hidden representation at the final layer
    MI_HX = entropy_input - entropy_input_h

    return nats2bits*MI_HX




