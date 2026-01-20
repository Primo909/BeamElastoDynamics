import numpy as np
def compute_von_mises_3d(stress_9):
    sxx = stress_9[:, 0]
    sxy = stress_9[:, 1]
    sxz = stress_9[:, 2]
    syx = stress_9[:, 3]
    syy = stress_9[:, 4]
    syz = stress_9[:, 5]
    szx = stress_9[:, 6]
    szy = stress_9[:, 7]
    szz = stress_9[:, 8]
    vm = 0.5 * (
        (sxx - syy) ** 2
        + (syy - szz) ** 2
        + (szz - sxx) ** 2
        + 6.0 * (sxy**2 + syz**2 + sxz**2)
    )
    return np.sqrt(vm)
