"""Signal processing functions built on MLX FFT primitives."""

import mlx.core as mx


def _validate_1d(x: mx.array, name: str):
    if x.ndim != 1:
        raise ValueError(f"{name} expects a 1-D array, got shape {x.shape}.")


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
    if n == 1:
        return mx.ones((1,))
    i = mx.arange(n)
    return 0.5 * (1.0 - mx.cos(2.0 * 3.141592653589793 * i / (n - 1)))


def _windowed_segments(x: mx.array, nperseg: int, step: int, n_segments: int) -> mx.array:
    """Return a strided (n_segments, nperseg) view over a 1-D signal."""
    x = mx.contiguous(x)
    return mx.as_strided(x, shape=(n_segments, nperseg), strides=(step, 1))


def _periodogram(x: mx.array, fs: float, window: bool) -> mx.array:
    """One-sided periodogram over the last axis; supports batched input."""
    n = x.shape[-1]
    if window:
        w = _hann_window(n)
        x = x * w
        s2 = mx.sum(w * w)
    else:
        s2 = mx.array(float(n))

    X = mx.fft.rfft(x, axis=-1)
    power = (mx.real(X) ** 2 + mx.imag(X) ** 2) / (fs * s2)
    power = power * 2.0

    if n % 2 == 0:
        return mx.concatenate(
            [power[..., :1] / 2.0, power[..., 1:-1], power[..., -1:] / 2.0],
            axis=-1,
        )
    return mx.concatenate([power[..., :1] / 2.0, power[..., 1:]], axis=-1)


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
    _validate_1d(x, "psd")
    n = x.shape[0]
    freqs = rfftfreq(n, d=1.0 / fs)
    power = _periodogram(x, fs=fs, window=window)
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
        nperseg: Length of each segment (must be > 0). Default: 256.
        noverlap: Number of overlapping samples (must be < nperseg). Default: nperseg // 2.

    Returns:
        (freqs, psd_estimate) averaged over segments.
    """
    _validate_1d(x, "welch")
    if nperseg <= 0:
        raise ValueError(f"nperseg must be > 0, got {nperseg}")
    if noverlap is None:
        noverlap = nperseg // 2
    if noverlap < 0 or noverlap >= nperseg:
        raise ValueError(f"noverlap must be in [0, nperseg), got {noverlap}")
    if x.shape[0] < nperseg:
        raise ValueError(f"Signal length ({x.shape[0]}) must be >= nperseg ({nperseg})")

    step = nperseg - noverlap
    n_segments = (x.shape[0] - nperseg) // step + 1

    if n_segments < 1:
        return psd(x, fs=fs, window=True)

    segments = _windowed_segments(x, nperseg, step, n_segments)
    powers = _periodogram(segments, fs=fs, window=True)
    freqs = rfftfreq(nperseg, d=1.0 / fs)
    return freqs, mx.mean(powers, axis=0)


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
        nperseg: Length of each segment (must be > 0).
        noverlap: Overlap between segments (must be < nperseg). Default: nperseg // 2.

    Returns:
        (times, freqs, Sxx) where Sxx has shape (n_segments, n_freqs).
    """
    _validate_1d(x, "spectrogram")
    if nperseg <= 0:
        raise ValueError(f"nperseg must be > 0, got {nperseg}")
    if noverlap is None:
        noverlap = nperseg // 2
    if noverlap < 0 or noverlap >= nperseg:
        raise ValueError(f"noverlap must be in [0, nperseg), got {noverlap}")
    if x.shape[0] < nperseg:
        raise ValueError(f"Signal length ({x.shape[0]}) must be >= nperseg ({nperseg})")

    step = nperseg - noverlap
    n_segments = (x.shape[0] - nperseg) // step + 1

    freqs = rfftfreq(nperseg, d=1.0 / fs)
    segments = _windowed_segments(x, nperseg, step, n_segments)
    Sxx = _periodogram(segments, fs=fs, window=True)
    times = (mx.arange(n_segments) * step + nperseg / 2) / fs
    return times, freqs, Sxx
