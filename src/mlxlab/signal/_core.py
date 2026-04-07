"""Signal processing functions built on MLX FFT primitives."""

import mlx.core as mx


def fftfreq(n: int, d: float = 1.0) -> mx.array:
    """Discrete Fourier transform sample frequencies.

    Equivalent to numpy.fft.fftfreq.

    Args:
        n: Window length.
        d: Sample spacing (inverse of sampling rate). Default: 1.0.

    Returns:
        Array of length n with frequencies: [0, 1, ..., n/2-1, -n/2, ..., -1] / (d*n)
    """
    val = 1.0 / (n * d)
    results = mx.zeros((n,))
    N = (n - 1) // 2 + 1
    p1 = mx.arange(0, N)
    p2 = mx.arange(-(n // 2), 0)
    freqs = mx.concatenate([p1, p2])
    return freqs * val


def rfftfreq(n: int, d: float = 1.0) -> mx.array:
    """Discrete Fourier transform sample frequencies (for rfft).

    Equivalent to numpy.fft.rfftfreq.

    Args:
        n: Window length.
        d: Sample spacing. Default: 1.0.

    Returns:
        Array of length n//2+1 with frequencies: [0, 1, ..., n/2] / (d*n)
    """
    val = 1.0 / (n * d)
    N = n // 2 + 1
    return mx.arange(0, N) * val


def _hann_window(n: int) -> mx.array:
    """Hann window."""
    i = mx.arange(n)
    return 0.5 * (1.0 - mx.cos(2.0 * 3.141592653589793 * i / (n - 1)))


def psd(x: mx.array, fs: float = 1.0, window: bool = True) -> tuple[mx.array, mx.array]:
    """Power spectral density via FFT.

    Args:
        x: 1-D signal array.
        fs: Sampling frequency. Default: 1.0.
        window: Apply Hann window. Default: True.

    Returns:
        (freqs, power) where freqs has length n//2+1 and power is the
        one-sided PSD in units^2/Hz.
    """
    n = x.shape[0]
    if window:
        w = _hann_window(n)
        x = x * w
        # Correct for window power
        S1 = float(mx.sum(w).item())
        S2 = float(mx.sum(w * w).item())
    else:
        S2 = float(n)

    X = mx.fft.rfft(x)
    power = (mx.real(X) ** 2 + mx.imag(X) ** 2) / (fs * S2)
    # Double one-sided (except DC and Nyquist)
    power = power * 2.0
    power = mx.concatenate([power[:1] / 2.0, power[1:-1], power[-1:] / 2.0])

    freqs = rfftfreq(n, d=1.0 / fs)
    return freqs, power


def welch(
    x: mx.array,
    fs: float = 1.0,
    nperseg: int = 256,
    noverlap: int | None = None,
) -> tuple[mx.array, mx.array]:
    """Welch's method for PSD estimation.

    Args:
        x: 1-D signal array.
        fs: Sampling frequency.
        nperseg: Length of each segment. Default: 256.
        noverlap: Number of overlapping samples. Default: nperseg // 2.

    Returns:
        (freqs, psd_estimate) averaged over segments.
    """
    if noverlap is None:
        noverlap = nperseg // 2

    step = nperseg - noverlap
    n_segments = (x.shape[0] - nperseg) // step + 1

    if n_segments < 1:
        return psd(x, fs=fs, window=True)

    power_sum = mx.zeros((nperseg // 2 + 1,))
    for i in range(n_segments):
        start = i * step
        segment = x[start : start + nperseg]
        _, p = psd(segment, fs=fs, window=True)
        power_sum = power_sum + p

    freqs = rfftfreq(nperseg, d=1.0 / fs)
    return freqs, power_sum / n_segments


def spectrogram(
    x: mx.array,
    fs: float = 1.0,
    nperseg: int = 256,
    noverlap: int | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Spectrogram via short-time Fourier transform.

    Args:
        x: 1-D signal array.
        fs: Sampling frequency.
        nperseg: Length of each segment.
        noverlap: Overlap between segments. Default: nperseg // 2.

    Returns:
        (times, freqs, Sxx) where Sxx has shape (n_segments, n_freqs).
    """
    if noverlap is None:
        noverlap = nperseg // 2

    step = nperseg - noverlap
    n_segments = (x.shape[0] - nperseg) // step + 1

    freqs = rfftfreq(nperseg, d=1.0 / fs)
    n_freqs = freqs.shape[0]

    segments = []
    times_list = []
    for i in range(n_segments):
        start = i * step
        segment = x[start : start + nperseg]
        _, p = psd(segment, fs=fs, window=True)
        segments.append(p)
        times_list.append((start + nperseg / 2) / fs)

    Sxx = mx.stack(segments)
    times = mx.array(times_list)
    return times, freqs, Sxx
