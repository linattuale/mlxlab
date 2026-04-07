"""Signal processing built on mlx.core.fft.

Provides utilities that MLX's FFT module doesn't ship:
- fftfreq, rfftfreq (frequency bin computation)
- psd (power spectral density)
- spectrogram (short-time Fourier transform)
- welch (Welch's method for PSD estimation)

For raw FFT operations (fft, ifft, rfft, irfft, fft2, etc.),
use mlx.core.fft directly.
"""

from ._core import fftfreq, rfftfreq, psd, spectrogram, welch

__all__ = ["fftfreq", "rfftfreq", "psd", "spectrogram", "welch"]
