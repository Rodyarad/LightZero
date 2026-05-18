import torch


def make_grid(images: torch.Tensor, nrow: int, pad_color: torch.Tensor = None) -> torch.Tensor:
    """Create a simple image grid from BCHW tensor."""
    if images.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {images.shape}")
    if nrow <= 0:
        raise ValueError(f"nrow must be positive, got {nrow}")

    b, c, h, w = images.shape
    ncols = nrow
    nrows = (b + ncols - 1) // ncols

    if pad_color is None:
        pad_color = torch.zeros(c, dtype=images.dtype, device=images.device)
    else:
        pad_color = pad_color.to(dtype=images.dtype, device=images.device)

    grid = pad_color.view(c, 1, 1).expand(c, nrows * h, ncols * w).clone()
    for idx in range(b):
        row = idx // ncols
        col = idx % ncols
        grid[:, row * h : (row + 1) * h, col * w : (col + 1) * w] = images[idx]
    return grid
