# NURBSDiff

A pure PyTorch implementation for differentiable NURBS curve and surface evaluation. This repository contains code for fitting NURBS curves and surfaces to any input point cloud using gradient-based optimization.

## Features

- **Pure PyTorch Implementation**: No C++/CUDA extensions required, works on any device
- **Differentiable NURBS Evaluation**: Fully differentiable curve and surface evaluation
- **Clean & Readable Code**: Uses [einops](https://github.com/arogozhnikov/einops) for clear tensor operations
- **Flexible**: Supports arbitrary degrees, control points, and knot vectors
- **GPU Accelerated**: Automatically utilizes GPU when available

## Installation

### Dependencies

1. **PyTorch**: Installation command can be generated from [here](https://pytorch.org/get-started/locally/)
2. **PyTorch3D** (for examples):
   - For CPU only: `pip install pytorch3d`
   - For macOS running on Apple Silicon: `MACOSX_DEPLOYMENT_TARGET=10.14 CC=clang CXX=clang++ pip install "git+https://github.com/facebookresearch/pytorch3d.git"`
   - For GPU support: `pip install "git+https://github.com/facebookresearch/pytorch3d.git"`
3. **Geomdl** (for examples): `pip install geomdl`
4. **einops**: `pip install einops`

### Install NURBSDiff

```bash
# Clone the repository
git clone https://github.com/your-repo/NURBSDiff.git
cd NURBSDiff

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

### NURBS Curve Evaluation

```python
import torch
from NURBSDiff.curve_eval import CurveEval

# Initialize curve evaluator
# m: number of control points - 1
# p: degree
# out_dim: number of evaluation points
curve_eval = CurveEval(m=10, dimension=3, p=2, out_dim=100, device='cuda')

# Control points: (batch_size, m+1, dimension+1)
# Last channel is the weight for rational curves
ctrl_pts = torch.rand(1, 11, 4).cuda()

# Evaluate curve
curve_points = curve_eval(ctrl_pts)  # (batch_size, out_dim, dimension)
```

### NURBS Surface Evaluation

```python
import torch
from NURBSDiff.surf_eval import SurfEval

# Initialize surface evaluator
surf_eval = SurfEval(
    m=10, n=10,           # control points - 1 in u, v directions
    dimension=3,
    p=3, q=3,             # degrees in u, v directions
    out_dim_u=128,
    out_dim_v=128,
    device='cuda'
)

# Control points: (batch_size, m+1, n+1, dimension+1)
ctrl_pts = torch.rand(1, 11, 11, 4).cuda()

# Evaluate surface
surface_points = surf_eval(ctrl_pts)  # (batch_size, out_dim_u, out_dim_v, dimension)
```

### Surface Fitting Example

```python
import torch
from NURBSDiff.surf_eval import SurfEval
from pytorch3d.loss import chamfer_distance

# Load or generate target point cloud
target_points = torch.rand(1, 1000, 3).cuda()

# Initialize NURBS surface
surf_eval = SurfEval(m=14, n=13, dimension=3, p=3, q=3,
                     out_dim_u=512, out_dim_v=512, device='cuda')

# Initialize control points as learnable parameters
ctrl_pts = torch.nn.Parameter(torch.rand(1, 15, 14, 4).cuda())

# Optimizer
optimizer = torch.optim.Adam([ctrl_pts], lr=0.01)

# Optimization loop
for iteration in range(1000):
    optimizer.zero_grad()

    # Evaluate NURBS surface
    surface_points = surf_eval(ctrl_pts)
    surface_points = surface_points.view(1, -1, 3)

    # Compute Chamfer distance
    loss, _ = chamfer_distance(surface_points, target_points)

    loss.backward()
    optimizer.step()

    if iteration % 100 == 0:
        print(f"Iteration {iteration}, Loss: {loss.item()}")
```

## Module Overview

### `curve_eval.py` - NURBS Curve Evaluation
Evaluates NURBS curves given control points and parameters.

**Input:**
- Control points: `(batch_size, m+1, dimension+1)` where last channel is weight

**Output:**
- Curve points: `(batch_size, out_dim, dimension)`

**Parameters:**
- `m`: Number of control points - 1
- `dimension`: Spatial dimension (2D or 3D)
- `p`: Curve degree
- `out_dim`: Number of evaluation points
- `knot_v`: Optional custom knot vector
- `device`: 'cpu' or 'cuda'

### `surf_eval.py` - NURBS Surface Evaluation
Evaluates NURBS surfaces with fixed knot vectors.

**Input:**
- Control points: `(batch_size, m+1, n+1, dimension+1)` where last channel is weight

**Output:**
- Surface points: `(batch_size, out_dim_u, out_dim_v, dimension)`

**Parameters:**
- `m`, `n`: Number of control points - 1 in u, v directions
- `dimension`: Spatial dimension
- `p`, `q`: Surface degrees in u, v directions
- `out_dim_u`, `out_dim_v`: Number of evaluation points
- `knot_u`, `knot_v`: Optional custom knot vectors
- `device`: 'cpu' or 'cuda'

### `nurbs_eval.py` - NURBS Surface with Learnable Knots
Evaluates NURBS surfaces with learnable/dynamic knot vectors (useful for optimization).

**Input:**
- Tuple of `(control_points, knot_u, knot_v)`
- Control points: `(batch_size, m+1, n+1, dimension)`
- Knot vectors: `(batch_size, knot_size)`

**Output:**
- Surface points: `(batch_size, out_dim_u, out_dim_v, dimension)`

### `utils.py` - Utility Functions
- `gen_knot_vector(p, n)`: Generate uniform knot vector
- `find_span_torch(n, p, u, U)`: Find knot span for parameter value
- `basis_funs_torch(i, u, p, U)`: Compute non-vanishing B-spline basis functions

## Migration from Previous Version

If you're upgrading from the C++/CUDA version, see [MIGRATION.md](MIGRATION.md) for a detailed guide.

**Quick changes:**
- Remove `method='tc'` and `dvc='cuda'` parameters
- Use `device='cuda'` or `device='cpu'` instead
- Control point tensors should be moved to device explicitly if needed

## Examples

See the `examples/` folder for complete examples:
- `NURBSSurfaceFitting.py` - Basic surface fitting
- `curve_fitting.py` - Curve fitting examples
- Various experiment files showing different configurations

## Testing

Run the test suite to verify correctness:

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_curve_eval.py
pytest tests/test_surf_eval.py
pytest tests/test_utils.py
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your_paper,
  title={Your Paper Title},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## License

[Add your license here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.