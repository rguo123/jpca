"""
jpca.py
=======
This file is for jpca functions.
"""

import numpy as np
from sklearn.decomposition import PCA
from math import sqrt
from scipy import optimize


def jpca(X, k=6):
    X_red, _ = pca_preproc(X, k)
    X_prestate = X_red[:-1,]
    # get discrete derivative of X_red w.r.t time (subtract adjacent rows)
    dX = np.diff(X_red, axis=0)
    m_skew = fit_skew(X_prestate, dX)
    M_skew = vec2vec2mat(m_skew)




def pca_preproc(X, k):
    pca = PCA(n_components=k)
    pca.fit(X)
    X_red = pca.transform(X)
    return X_red, pca.explained_variance_ratio_


def fit_skew(X_prestate, dX):
    # guaranteed to be square
    M0, _, _, _ = np.linalg.lstsq(X_prestate, dX, rcond=None)
    M0_skew = .5*(M0 - M0.T)
    m_skew = mat2vec(M0_skew)
    opt = optimize_skew(m_skew, X_prestate, dX)
    return opt.x


def optimize_skew(m_skew, X_prestate, dX):
    def objective(x, X_prestate, dX):
        f = np.linalg.norm(dX - X_prestate@vec2mat(x))
        return f**2

    def derivative(x, X_prestate, dX):
        D = dX - X_prestate@vec2mat(x)
        D = D.T @ X_prestate
        return 2*mat2vec(D - D.T)

    return optimize.minimize(objective, m_skew,
                                        jac=derivative,
                                        args=(X_prestate, dX))


def mat2vec(mat):
    return mat.flatten('F')


def vec2mat(vec):
    shape = (int(sqrt(vec.size)), -1)
    return np.reshape(vec, shape, 'F')
