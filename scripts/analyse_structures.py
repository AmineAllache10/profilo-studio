import numpy as np

def frequence_sillons(z):
    profil = np.nanmean(z, axis=0)
    profil = profil - np.nanmean(profil)

    F = np.abs(np.fft.rfft(profil))
    freqs = np.fft.rfftfreq(len(profil))

    idx = np.argmax(F[1:]) + 1
    return freqs[idx]
