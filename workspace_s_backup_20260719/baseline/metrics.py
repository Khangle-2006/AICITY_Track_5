from __future__ import annotations

import math


def psnr(cv2, np, pred, target) -> float:
    pred_f = pred.astype(np.float32) / 255.0
    target_f = target.astype(np.float32) / 255.0
    mse = float(np.mean((pred_f - target_f) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def ssim_gray(cv2, np, pred, target) -> float:
    p = cv2.cvtColor(pred, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
    t = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
    c1 = 0.01**2
    c2 = 0.03**2
    mu_p = p.mean()
    mu_t = t.mean()
    var_p = p.var()
    var_t = t.var()
    cov = ((p - mu_p) * (t - mu_t)).mean()
    return float(((2 * mu_p * mu_t + c1) * (2 * cov + c2)) / ((mu_p**2 + mu_t**2 + c1) * (var_p + var_t + c2)))


def mean_absdiff(np, pred, target) -> float:
    return float(np.mean(np.abs(pred.astype(np.float32) - target.astype(np.float32))))
