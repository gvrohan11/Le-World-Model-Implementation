import torch

def sigreg_loss(z, n_directions=1024, n_knots=17):
    if z.ndim == 2:
        z = z.unsqueeze(1)

    projected_input = z.transpose(0,1)
    batch_size  = projected_input.shape[1]
    dim = projected_input.shape[2]

    directions = torch.randn(dim, n_directions, device=z.device, dtype=z.dtype)
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)
    t = torch.linspace(0.0, 3.0, n_knots, device=z.device, dtype=z.dtype)
    dt = 3.0 / (n_knots - 1)

    weights = torch.full_like(t, 2.0 * dt)
    weights[0] = dt
    weights[-1] = dt

    gaussian = torch.exp(-0.5 * t.square())
    weights *= gaussian

    projection = (projected_input @ directions).unsqueeze(-1) * t
    error = (projection.cos().mean(dim=-3) - gaussian).square() + projection.sin().mean(dim=-3).square()
    statistic = (error @ weights) * batch_size
    return statistic.mean()