import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

np.random.seed(1)

# Smaller grid 30x30
nx, ny = 30, 30
nmodel = nx * ny
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
xx, yy = np.meshgrid(x, y, indexing='xy')

# True model: localized negative anomalies
true_model = np.zeros((ny, nx))
cx1, cy1, r1, amp1 = 0.35, 0.45, 0.08, -0.25
mask1 = ((xx - cx1)**2 + (yy - cy1)**2) <= r1**2
true_model[mask1] = amp1
cx2, cy2, r2, amp2 = 0.7, 0.6, 0.06, -0.18
mask2 = ((xx - cx2)**2 + (yy - cy2)**2) <= r2**2
true_model[mask2] = amp2
m_true = true_model.ravel()

# Observations
nobs = 40
obs_x = np.linspace(0.05, 0.95, nobs)
obs_y = np.zeros_like(obs_x) + 1.02
obs_locations = np.vstack([obs_x, obs_y]).T

def build_G(obs_locations, xx, yy, depth_scale=0.08):
    nobs = obs_locations.shape[0]
    nx, ny = xx.shape[1], xx.shape[0]
    nmodel = nx * ny
    G = np.zeros((nobs, nmodel))
    for i, (ox, oy) in enumerate(obs_locations):
        dx = xx - ox
        dy = yy - oy
        dist2 = dx**2 + dy**2 + (depth_scale**2)
        G[i, :] = 1.0 / (dist2.ravel())
    G = G / G.sum(axis=1, keepdims=True)
    return G

G = build_G(obs_locations, xx, yy, depth_scale=0.08)
d_clean = G.dot(m_true)
noise = np.random.normal(0, 0.01, size=d_clean.shape)
d_obs = d_clean + noise

# Laplacian
def laplacian_2d(nx, ny):
    n = nx * ny
    rows = []
    cols = []
    data = []
    def idx(i,j):
        return j*nx + i
    for j in range(ny):
        for i in range(nx):
            k = idx(i,j)
            rows.append(k); cols.append(k); data.append(4.0)
            if i > 0:
                rows.append(k); cols.append(idx(i-1,j)); data.append(-1.0)
            if i < nx-1:
                rows.append(k); cols.append(idx(i+1,j)); data.append(-1.0)
            if j > 0:
                rows.append(k); cols.append(idx(i,j-1)); data.append(-1.0)
            if j < ny-1:
                rows.append(k); cols.append(idx(i,j+1)); data.append(-1.0)
    L = sparse.csr_matrix((data, (rows, cols)), shape=(n,n))
    return L

L = laplacian_2d(nx, ny)

GTG = G.T.dot(G)
GTD = G.T.dot(d_obs)

lambdas = [1e-6, 1e-4, 5e-3]
solutions = []
for lam in lambdas:
    A = GTG + lam * (L.T.dot(L)).toarray()
    m_est = np.linalg.solve(A, GTD)
    solutions.append(m_est)

# Plot
fig = plt.figure(figsize=(10,6))
plt.subplot(2,3,1)
plt.title('True model (density contrast proxy)')
plt.imshow(true_model, origin='lower', extent=[0,1,0,1])
plt.xlabel('x'); plt.ylabel('y')
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(2,3,2)
plt.title('Observed data (synthetic)')
plt.plot(obs_x, d_obs, marker='o')
plt.xlabel('Observation x'); plt.ylabel('Observed signal')

plt.subplot(2,3,3)
plt.title('Clean data (noise-free)')
plt.plot(obs_x, d_clean, marker='o')
plt.xlabel('Observation x'); plt.ylabel('Signal')

for i, lam in enumerate(lambdas):
    plt.subplot(2,3,4+i)
    plt.title(f'Recovered model (lambda={lam:g})')
    plt.imshow(solutions[i].reshape((ny,nx)), origin='lower', extent=[0,1,0,1])
    plt.xlabel('x'); plt.ylabel('y')
    plt.colorbar(fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

for lam, m_est in zip(lambdas, solutions):
    residual = G.dot(m_est) - d_obs
    misfit = np.sqrt(np.mean(residual**2))
    model_norm = np.linalg.norm(L.dot(m_est))
    print(f"lambda={lam:.1e}  RMS misfit={misfit:.4e}  model_smoothness_norm={model_norm:.4e}")