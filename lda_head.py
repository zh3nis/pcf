import torch
import torch.nn as nn
import torch.nn.functional as F


class LDAHead(nn.Module):
    """LDA head with trainable class means, shared variance, and optional trainable priors.

    Class-conditionals are Gaussian with a shared trainable variance.
    """

    def __init__(self, C: int, D: int, fixed_variance: float = 1.0, train_priors: bool = True):
        super().__init__()
        if C < 2:
            raise ValueError(f"C must be at least 2 (got C={C}).")
        if D < 1:
            raise ValueError(f"D must be positive (got D={D}).")
        if fixed_variance <= 0:
            raise ValueError(f"fixed_variance must be > 0 (got {fixed_variance}).")

        self.C = C
        self.D = D
        dtype = torch.get_default_dtype()
        self.log_variance = nn.Parameter(torch.log(torch.tensor(float(fixed_variance), dtype=dtype)))

        self.mu = nn.Parameter(torch.zeros(C, D, dtype=dtype))

        if train_priors:
            self.prior_logits = nn.Parameter(torch.zeros(C, dtype=dtype))
        else:
            self.register_buffer("prior_logits", torch.zeros(C, dtype=dtype))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[1] != self.D:
            raise ValueError(f"Expected z shape (N, {self.D}), got {tuple(z.shape)}")

        mu = self.mu.to(dtype=z.dtype, device=z.device)
        prior_logits = self.prior_logits.to(dtype=z.dtype, device=z.device)
        var = F.softplus(self.log_variance).to(dtype=z.dtype, device=z.device)

        diff = z.unsqueeze(1) - mu.unsqueeze(0)  # (N, C, D)
        m2 = (diff * diff).sum(dim=-1)  # (N, C)
        log_prior = F.log_softmax(prior_logits, dim=0)  # (C,)

        # log p(z | y=c) + log p(y=c), including Gaussian normalization constant.
        log_norm = -0.5 * self.D * torch.log(2.0 * torch.pi * var)
        return log_prior.unsqueeze(0) + log_norm - 0.5 * m2 / var

    @torch.no_grad()
    def priors(self) -> torch.Tensor:
        return torch.softmax(self.prior_logits, dim=0)
