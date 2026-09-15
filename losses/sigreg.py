import torch

def sigreg_loss(Z, n_directions=128, t_max=5.0, n_knots=33):
    B, D = Z.shape

    V = torch.randn(D, n_directions, device=Z.device)

    V = V / V.norm(dim=0, keepdim=True)

    P = Z @ V

    t = torch.linspace(-t_max, t_max, n_knots, device = Z.device)

    tP = t[:, None, None] * P[None, : , :]
    Re = torch.cos(tP).mean(dim=1)
    Im = torch.sin(tP).mean(dim=1)
    phi = torch.exp(-0.5 * t**2)
    diff2 = (Re - phi[:, None])**2 + Im**2

    w = torch.exp(-0.5 * t**2)
    dt = t[1] - t[0]
    stat = (diff2 * w[:, None]).sum(dim=0) * dt
    return stat.mean()

if __name__ == "__main__":
    torch.manual_seed(0)
    print("Gaussian cloud: ", round(sigreg_loss(torch.randn(512, 32)).item(), 4))
    print("Collapsed blob: ", round(sigreg_loss(torch.ones(512, 32) * 0.3).item(), 4))
    print("Wrong-scale cloud: ", round(sigreg_loss(torch.randn(512, 32) * 4).item(), 4))

    Z = (torch.ones(512, 32) + 0.01 * torch.randn(512, 32)).requires_grad_(True)
    opt = torch.optim.Adam([Z], lr=0.5)
    for _ in range(400):
        opt.zero_grad()
        L = sigreg_loss(Z)
        L.backward()
        opt.step()
    print(f"Near-collapsed blob after training -> std: {round(Z.std().item(), 3)} (want ~1.0)")