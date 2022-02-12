import numpy as np
import matplotlib.pyplot as plt



class UnscentedKalmanFilter(object):
  """
  Unscented Kalman Filter
  """
  def __init__(self, n_landmarks=10, dt=1):
    self.n_ld = n_ld
    self.n_mu = 3 + 2*n_ld
    self.mu = np.zeros((self.n_mu, 1))
    self.S = np.eye(self.n_mu, dtype=np.float32) * 1e8
    self.S[:3, :3] = 0

  def update(self, u, zm):
    # UKF initialisation
    kapa, alpha, beta, n_mu, n_ld = 5, 0.4, 2, self.n_mu, self.n_ld
    lamb = (n_mu+kapa)*(alpha**2) - n_mu
    L = np.linalg.cholesky(self.S)
    
    # Sigma points
    X = np.repeat(self.mu, (2*n_mu)+1, axis=1)
    X[:, 1:n_mu+1] += np.sqrt(lamb+n_mu) * L
    X[:, n_mu+1:] -= np.sqrt(lamb+n_mu) * L
  
    # Weights
    wm = np.repeat(1/(2*(n+lamb)), 2*n_mu+1, axis=0)
    wm[0] *= 2 * lamb
    wc = np.copy(wm)
    wc[0] += (1 - alpha**2 + beta)
    wc.reshape(1, -1)
    wm.reshape(1, -1)

    # Predict the new mean
    Xp = np.copy(X)

    R = np.zeros_like(self.S)
    Q = np.zeros_like(self.S)
    dtheta = ut[1]*dt

    # Update the mean vector
    if ut[1] != 0:
      radius = ut[0]/ut[1]
      Xp[:3] += np.array([(-np.sin(X[2:3]) + np.sin(X[2:3]+dtheta)) * radius,
                          (+np.cos(X[2:3]) - np.cos(X[2:3]+dtheta)) * radius,
                          np.repeat(np.array([[dtheta]]), 2*n_mu+1, axis=1)], axis=0)

    else:
      Xp[:3] += np.array([np.cos(X[2:3]) * ut[0],
                          np.sin(X[2:3]) * ut[0],
                          np.repeat(np.array([[0]]), 2*n_mu+1, axis=1)], axis=0)
    # compute the the mean estimate and the covariance estimate
    self.mu = np.sum(wm * Xp, axis=1, keepdims=True)
    Xp -= self.mu
    self.S = np.zeros_like(self.S)
    for i in range(2*n_mu+1):
      self.S += wc[i] * Xp[:,i:i+1] @ Xp[:,i:i+1].T
    self.S += R

    # Update the sigma points for observation
    L = np.linalg.cholesky(self.S)
    X = np.repeat(self.mu, (2*n_mu)+1, axis=1)
    X[:, 1:n_mu+1] += np.sqrt(lamb+n_mu) * L
    X[:, n_mu+1:] -= np.sqrt(lamb+n_mu) * L
    dX = X - self.mu

    # Predict the position of the landmarks
    LX = np.reshape(X[3:].T, (-1, n_ld, 2))
    DX = LX - np.reshape(X[:2].T, (-1, 1, 2))
    RX2 = np.sum(DX**2, axis=2)
    RX = np.sqrt(RX2).T
    ThetaX = np.arctan2(DX[:, :, 1], DX[:, :, 0]).T - X[2:3, :]
    Zp = np.concatenate([RX, ThetaX], axis=0)

    # Compute the mean predicted measurements
    zp_mean = np.sum(wm * Zp, axis=1, keepdims=True)
    dZ = Zp - zp_mean
    S = Q
    for i in range(2*n_mu+1):
      S += wc[i] * dZ[:, i:i+1] @ dZ[:, i:i+1].T

    Sigma = np.zeros((n_mu, 2*n_mu+1))
    for i in range(2*n_mu+1):
      Sigma += wc[i] * dX[:, i:i+1] @ dZ[:, i:i+1].T

    K = Sigma @ np.linalg.inv(S)
    self.mu +=  K @ (zm - zp_mean)
    self.S -= K @ S @ K.T

