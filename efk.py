import numpy as np
import matplotlib.pyplot as plt



n = 40
params = 3 + 2 * n
mu = np.zeros((params, 1), dtype=np.float32)
mr = np.zeros((3, 1), dtype=np.float32)
S = np.eye(params, dtype=np.float32) * 1e8
S[:3, :3] = 0
rldms = np.random.randint(-100, 100, size=(n, 2))


mu[3:] = np.random.randint(-100, 100, size=mu[3:].shape)

pldms = None



def extended_kalman_filter(ut, dt=0.1):
  global  mu, mr, S, rldms, pldms, params
  R = np.diag([0, 0, 0])
  Qu = np.diag([1, 1e-1])
  dtheta = ut[1]*dt
  # update the mean vector
  if ut[1] != 0:
    radius = ut[0]/ut[1]
    mu[:3, :1] += np.array([(-np.sin(mu[2]) + np.sin(mu[2]+dtheta)) * radius,
                            (+np.cos(mu[2]) - np.cos(mu[2]+dtheta)) * radius,
                            dtheta], dtype=np.float32)
    mr += np.array([(-np.sin(mr[2]) + np.sin(mr[2]+dtheta)) * radius,
                            (+np.cos(mr[2]) - np.cos(mr[2]+dtheta)) * radius,
                            dtheta], dtype=np.float32)

  else:
    mu[:3, :1] += np.array([np.cos(mu[2]) * ut[0],
                            np.sin(mu[2]) * ut[0],
                            [0]], dtype=np.float32)

    mr += np.array([np.cos(mr[2]) * ut[0],
                            np.sin(mr[2]) * ut[0],
                            [0]], dtype=np.float32)
  # update the covariance matrix
  G = np.eye(3, dtype=np.float32)
  if ut[1] != 0:
    radius = ut[0]/ut[1]
    G[:2, 2:3] = np.array([(-np.cos(mu[2]) + np.cos(mu[2]+dtheta)) * radius,
                         (-np.sin(mu[2]) + np.sin(mu[2]+dtheta)) * radius], dtype=np.float32)
  else:
    G[:2, 2:3] = np.array([-np.sin(mu[2]) * ut[0],
                           +np.cos(mu[2]) * ut[0]], dtype=np.float32)

  S[:3, :3] = G @ S[:3, :3] @ G.T + R
  S[:3, 3:] = G @ S[:3, 3:]
  S[3:, :3] = S[:3, 3:].T
  rdl = rldms - mu[:2].T
  rdist2 = rdl[:, :1]**2 + rdl[:, 1:]**2
  rdist = np.sqrt(rdist2)
  rtheta = np.arctan2(rdl[:, 1:], rdl[:, :1]) - mu[2][0]
  mdist = rdist # + np.random.rand(*rdist.shape)  * Qu[0, 0]
  mtheta = rtheta # + np.random.rand(*rdist.shape) * Qu[1, 1]


  max_dist = 100
  closed_m = np.argwhere(mdist.flatten() < max_dist).flatten()
  pldms = mu[3:].reshape(-1, 2)
  pdl = pldms - mu[:2].T
  pdist2 = pdl[:, :1]**2 + pdl[:, 1:]**2
  pdist = np.sqrt(pdist2)
  ptheta = np.arctan2(pdl[:, 1:], pdl[:, :1]) - mu[2][0]
  zm = np.concatenate([mdist[closed_m], mtheta[closed_m]], axis=1).reshape(-1, 1)
  zp = np.concatenate([pdist[closed_m], ptheta[closed_m]], axis=1).reshape(-1, 1)

  H = np.zeros((2*closed_m.shape[0], params))
  for (i, mi) in enumerate(closed_m):
    H[i*2:(i+1)*2, :3] = np.array([[-pdl[mi,0]/pdist[mi,0],  -pdl[mi,1]/pdist[mi, 0],  0],
                                   [pdl[mi,1]/pdist2[mi,0], -pdl[mi,0]/pdist2[mi, 0], -1]])
    H[i*2:(i+1)*2, 3+(mi*2):3+((mi+1)*2)] = np.array([[  pdl[mi,0]/pdist[mi,0],  pdl[mi,1]/pdist[mi,0]],
                                                      [-pdl[mi,1]/pdist2[mi,0], pdl[mi,0]/pdist2[mi,0]]])
  Q = np.eye(2*closed_m.shape[0])
  K = S @ H.T @ np.linalg.inv(H @ S @ H.T + Q)
  mu = mu + K @ (zm - zp)
  S = (np.eye(params, dtype=np.float32) - K @ H) @ S
  return None



def main():
  global ldms, mu, S
  fig = plt.figure()
  ax1 = fig.add_subplot(121)
  ax2 = fig.add_subplot(122)
  plt.show(block=False)
  dt = 1
  vel = 2
  r = 60
  rot = 100
  rot_t = int(2*rot*r*np.pi / (vel*dt))
  vel_cmds = np.concatenate([np.repeat(vel, int(r/(dt*vel))).reshape(-1, 1), np.array([[0]]),
                            np.repeat(vel, rot_t).reshape(-1, 1)], axis=0)
  omg_cmds = np.concatenate([np.repeat(0, int(r/(dt*vel))).reshape(-1, 1), np.array([[np.pi/2]]),
                            np.repeat(2*rot*np.pi/rot_t, rot_t).reshape(-1, 1)], axis=0)
  for (i, u) in enumerate(zip(vel_cmds, omg_cmds)):
    extended_kalman_filter(u, dt)
    if i & 63 == 0:
      ax1.clear()
      ax1.scatter(rldms[:, 0], rldms[:, 1], color="red")
      ax1.scatter(pldms[:, 0], pldms[:, 1], color="blue")
      ax2.scatter(mu[0], mu[1], color="red")
      ax2.scatter(mr[0], mr[1], color="blue")
      fig.canvas.draw()
      fig.canvas.flush_events()
    if i & 63 == 0:
      S = np.eye(params, dtype=np.float32) * 1e10
      S[:3, :3] = 0


if __name__ == '__main__':
  main()
