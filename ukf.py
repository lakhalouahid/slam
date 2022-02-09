import numpy as np
import matplotlib.pyplot as plt



class UnscentedKalmanFilter(object):
  """
  Unscented Kalman Filter
  """
  def __init__(self):
    n = 3 + (2*10)
    self.mu = np.zeros((n, 1))
    self.S = np.eye(n, dtype=np.float32) * 1e8
    self.S[:3, :3] = 0

  def update(self, u, zm):
    ## UKF initialisation
    kapa, alpha, beta, n = 5, 0.4, 2, self.S.shape[0]
    lamb = (n+kapa)*alpha**2 - n
    L = np.linalg.cholesky(self.S)
    
    X = np.zeros((n, (2*n)+1))
    X[:, 0] = self.mu
    X[:, 1:n+1] = self.mu + np.sqrt(lamb+n) * L
    X[:, n+1:] = self.mu - np.sqrt(lamb+n) * L

    
