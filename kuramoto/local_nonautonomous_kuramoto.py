# %%
from imports import *
import matplotlib as mpl

from scipy.signal import convolve2d
from scipy.signal.windows import gaussian

# %%
# https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html
mpl.rcParams["image.interpolation"] = "none"
# %%

np.random.seed(0)
n = 100
N = n ** 2
K = 1  # np.sqrt(np.pi / 4)
ω = np.random.rand(N) * 2 * np.pi
_ω = ω.reshape(n, n)
θ = np.random.rand(N) * 2 * np.pi
α = np.random.randn(N)
_α = α.reshape(n, n)
# %%
# _ω = ω.reshape(n, n)
# _θ = θ.reshape(n, n)
# _θ
# %%
# kernel = np.full((5, 5), 1 / 9) + np.diag([i for i in range(5)])
k_size = 17


window = gaussian(k_size, std=k_size / np.sqrt(n))
plt.plot(window)
# %%
kernel = np.atleast_2d(window).T * window
# %%

plt.imshow(kernel)
# %%


def F(t, θ):

    # dθ = dθ.reshape(n, n)
    _θ = θ.reshape(n, n)
    dθ = np.zeros_like(_θ)

    def f(θ_i, θ_j, t):
        return np.sin(θ_j - θ_i) + np.sin(t * (θ_j - θ_i) ** 2) / 2

    for i in range(n):
        for j in range(n):
            # print(_θ[i, j])
            phase_differences = f(_θ[i, j], _θ, t)
            # _θ[i, j] = 1
            dθ[i, j] = _ω[i, j] * np.cos(t) + (
                K
                * _α[i, j]
                * np.sum(convolve2d(phase_differences, kernel))
                / k_size ** 2
            )
    return dθ.flatten()


# %%


integrator = Integrators["ForwardEuler"]()
# %%

ts, θs = integrator.solve(F, 0, 15, θ, 0.1)
NUM_TS = len(ts)
θs = θs.reshape(NUM_TS, n, n)
# %%
np.min(θs)
np.max(θs)

# %%
fig, ax = plt.subplots(figsize=(n // 10, n // 10))
ax.set_axis_off()
im = ax.imshow(θs[0])  # , vmin=0, vmax=2 * np.pi)
# fig.colorbar(im)


def init_plot():
    return ax.images


def update(num, θs, ax):
    ax.images[0].set_data(θs[num])
    return ax.images


anim = animation.FuncAnimation(
    fig,
    update,
    frames=NUM_TS,
    fargs=(θs, ax),
    interval=5,
    blit=True,
)
# %%

file_path = os.path.join(KURAMOTO_OUTS, f"local_nonautonomous_kuramoto_K={K:.4f}.mp4")
anim.save(file_path, fps=6)
