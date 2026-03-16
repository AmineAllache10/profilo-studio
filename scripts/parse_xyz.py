import numpy as np

def lire_xyz(filepath):
    with open(filepath) as f:
        lines = f.readlines()

    start = next(i for i, l in enumerate(lines) if l.strip().startswith("#")) + 1
    end = len(lines) - next(i for i, l in enumerate(reversed(lines)) if l.strip().startswith("#")) - 1

    data = []
    for line in lines[start:end]:
        parts = line.split()
        if len(parts) == 3 and parts[2] != "No":
            x, y, z = map(float, parts)
            data.append((x, y, z))

    return np.array(data)

def creer_grille(points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    nx, ny = int(x.max()) + 1, int(y.max()) + 1

    Z = np.full((ny, nx), np.nan)
    for xi, yi, zi in points:
        Z[int(yi), int(xi)] = zi

    return Z
