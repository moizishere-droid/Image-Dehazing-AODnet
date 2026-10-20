"""RGB PSNR and canonical Gaussian-window SSIM, with explicit conventions."""
import math
import torch
from torch.nn import functional as F

def psnr(prediction, reference):
    mse = torch.mean((prediction.double() - reference.double()) ** 2).item()
    return float("inf") if mse == 0 else 10 * math.log10(1 / mse)

def ssim(prediction, reference):
    if min(prediction.shape[-2:]) < 11:
        raise ValueError("SSIM needs images at least 11 x 11 pixels")
    x, y = prediction.double(), reference.double()
    g = torch.arange(-5, 6, dtype=torch.float64, device=x.device)
    g = torch.exp(-g.square() / (2 * 1.5 ** 2)); g /= g.sum()
    kernel = torch.outer(g, g)[None, None].repeat(3, 1, 1, 1)
    avg = lambda v: F.conv2d(v, kernel, groups=3)
    ux, uy = avg(x), avg(y)
    vx, vy = avg(x*x) - ux*ux, avg(y*y) - uy*uy
    cov = avg(x*y) - ux*uy
    score = ((2*ux*uy + .01**2) * (2*cov + .03**2) /
             ((ux*ux + uy*uy + .01**2) * (vx + vy + .03**2)))
    return score.mean().item()

def scores(prediction, reference):
    return {"psnr_db": psnr(prediction, reference), "ssim": ssim(prediction, reference)}
