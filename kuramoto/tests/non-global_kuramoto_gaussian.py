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
n = 30
N = n ** 2
K = 1  # np.sqrt(np.pi)
ω = np.random.rand(N) * 2 * np.pi
_ω = ω.reshape(n, n)
θ = np.random.rand(N) * 2 * np.pi
# %%
# _ω = ω.reshape(n, n)
_θ = θ.reshape(n, n)
# _θ
# %%
# kernel = np.full((5, 5), 1 / 9) + np.diag([i for i in range(5)])
k_size1 = 9
k_size2 = k_size1 // 2 + 1
k_size2
window1 = gaussian(k_size1, std=np.sqrt(K) * k_size1 / np.sqrt(n))
window2 = gaussian(k_size2, std=np.sqrt(K) * k_size2 / np.sqrt(n))
plt.plot(window1)
plt.plot(window2)
window = np.convolve(np.fft.fftshift(window1), window2)
plt.plot(window)

window_ = np.convolve(np.fft.fftshift(window1), np.fft.fftshift(window2))
plt.plot(window_)


# %%
kernel = np.atleast_2d(window).T * window
plt.imshow(kernel)
kernel_ = np.atleast_2d(window_).T * window_
plt.imshow(kernel_)
plt.imshow(convolve2d(kernel, kernel_))
# %%
kernel = convolve2d(kernel, kernel_)


def F(t, θ):

    # dθ = dθ.reshape(n, n)
    _θ = θ.reshape(n, n)
    dθ = np.zeros_like(_θ)
    f = lambda θ_i, θ_j: np.sin(θ_j - θ_i)
    for i in range(n):
        for j in range(n):
            # print(_θ[i, j])
            # dθ[i] = ω[i] + (K / N) * np.sum(np.sin(θ - θ_i))
            phase_difference = f(_θ[i, j], _θ)
            conv = convolve2d(phase_difference, kernel)
            dθ[i, j] = _ω[i, j] + K * np.sum(conv)
    return dθ.flatten()


# %%


integrator = Integrators["ForwardEuler"]()
# %%

ts, θs = integrator.solve(F, 0, 2, θ, 0.1)
NUM_TS = len(ts)
θs = θs.reshape(NUM_TS, n, n)
# %%


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

file_path = os.path.join(KURAMOTO_OUTS, f"nonglobal_kuramoto_gaussian_K={K:.4f}.mp4")
anim.save(file_path, fps=6)
