# Migration Guide: v1.x to v2.0

This guide helps you migrate from the C++/CUDA-based version (v1.x) to the pure PyTorch implementation (v2.0).

## Overview of Changes

### What Changed?
- **Removed C++/CUDA Extensions**: All C++/CUDA code has been replaced with pure PyTorch
- **Simplified API**: Cleaner interface using standard PyTorch conventions
- **Added einops**: Using einops for clearer tensor operations
- **Device Management**: Consistent device handling using PyTorch conventions

### What Stayed the Same?
- **Core NURBS Algorithm**: The mathematical implementation remains identical
- **API Structure**: Module names and basic usage patterns are preserved
- **Accuracy**: Numerical results should match the previous version

## API Changes

### CurveEval

**Old (v1.x):**
```python
from NURBSDiff.curve_eval import CurveEval

curve_eval = CurveEval(
    m=10,
    dimension=3,
    p=2,
    out_dim=100,
    method='tc',      # ❌ Removed
    dvc='cuda'        # ❌ Removed
)
```

**New (v2.0):**
```python
from NURBSDiff.curve_eval import CurveEval

curve_eval = CurveEval(
    m=10,
    dimension=3,
    p=2,
    out_dim=100,
    device='cuda'     # ✅ New parameter
)
```

### SurfEval

**Old (v1.x):**
```python
from NURBSDiff.surf_eval import SurfEval

surf_eval = SurfEval(
    m=10, n=10,
    dimension=3,
    p=3, q=3,
    out_dim_u=128,
    out_dim_v=128,
    method='tc',      # ❌ Removed
    dvc='cuda'        # ❌ Removed
).cuda()              # ❌ No longer needed
```

**New (v2.0):**
```python
from NURBSDiff.surf_eval import SurfEval

surf_eval = SurfEval(
    m=10, n=10,
    dimension=3,
    p=3, q=3,
    out_dim_u=128,
    out_dim_v=128,
    device='cuda'     # ✅ New parameter
)
```

### NURBS Surface with Learnable Knots

**Old (v1.x):**
```python
from NURBSDiff.nurbs_eval import SurfEval

surf_eval = SurfEval(
    m=10, n=10,
    dimension=3,
    p=3, q=3,
    out_dim_u=128,
    out_dim_v=128,
    method='tc',      # ❌ Removed
    dvc='cpp'         # ❌ Removed
)
```

**New (v2.0):**
```python
from NURBSDiff.nurbs_eval import SurfEval

surf_eval = SurfEval(
    m=10, n=10,
    dimension=3,
    p=3, q=3,
    out_dim_u=128,
    out_dim_v=128,
    device='cuda'     # ✅ New parameter
)
```

## Step-by-Step Migration

### 1. Update Dependencies

**Old requirements:**
- PyTorch with C++ extensions
- CUDA toolkit (if using GPU)
- C++ compiler

**New requirements:**
```bash
pip install torch einops numpy
```

### 2. Update Installation

**Old:**
```bash
python setup.py develop  # Compiled C++/CUDA extensions
```

**New:**
```bash
pip install -e .  # Pure Python installation
```

### 3. Update Your Code

Search and replace in your codebase:

```python
# Replace
method='tc', dvc='cuda'
# With
device='cuda'

# Replace
method='tc', dvc='cpu'
# With
device='cpu'

# Remove redundant .cuda() calls
surf_eval = SurfEval(...).cuda()  # Old
surf_eval = SurfEval(..., device='cuda')  # New
```

### 4. Device Management

**Old approach:**
```python
# Initialize on CPU, then move to GPU
surf_eval = SurfEval(m=10, n=10, p=3, q=3, out_dim_u=128, out_dim_v=128)
surf_eval = surf_eval.cuda()

# Control points needed explicit device management
ctrl_pts = torch.rand(1, 11, 11, 4)
ctrl_pts = ctrl_pts.cuda()
```

**New approach:**
```python
# Initialize directly on target device
surf_eval = SurfEval(m=10, n=10, p=3, q=3, out_dim_u=128, out_dim_v=128, device='cuda')

# Control points still need device management
ctrl_pts = torch.rand(1, 11, 11, 4, device='cuda')
# or
ctrl_pts = torch.rand(1, 11, 11, 4).cuda()
```

## Common Migration Patterns

### Pattern 1: Basic Surface Fitting

**Old:**
```python
layer = SurfEval(14, 13, dimension=3, p=3, q=3,
                 out_dim_u=512, out_dim_v=512,
                 method='tc', dvc='cuda').cuda()

ctrl_pts = torch.nn.Parameter(torch.rand(1, 15, 14, 4)).cuda()
```

**New:**
```python
layer = SurfEval(14, 13, dimension=3, p=3, q=3,
                 out_dim_u=512, out_dim_v=512,
                 device='cuda')

ctrl_pts = torch.nn.Parameter(torch.rand(1, 15, 14, 4, device='cuda'))
```

### Pattern 2: With Custom Knot Vectors

**Old:**
```python
layer = SurfEval(14, 13, knot_u=knot_u, knot_v=knot_v,
                 dimension=3, p=3, q=3,
                 out_dim_u=512, out_dim_v=512,
                 method='tc', dvc='cuda').cuda()
```

**New:**
```python
layer = SurfEval(14, 13, knot_u=knot_u, knot_v=knot_v,
                 dimension=3, p=3, q=3,
                 out_dim_u=512, out_dim_v=512,
                 device='cuda')
```

### Pattern 3: Dynamic Device Selection

**Old:**
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
layer = SurfEval(10, 10, p=3, q=3, out_dim_u=128, out_dim_v=128,
                 method='tc', dvc=device)
if device == 'cuda':
    layer = layer.cuda()
```

**New:**
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
layer = SurfEval(10, 10, p=3, q=3, out_dim_u=128, out_dim_v=128,
                 device=device)
```

## Automated Migration Script

Use this Python script to automatically update your code:

```python
import re
import sys

def migrate_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Replace method and dvc parameters with device
    content = re.sub(
        r"method='tc',\s*dvc='(cuda|cpu)'",
        r"device='\1'",
        content
    )

    # Remove redundant .cuda() calls after SurfEval/CurveEval
    content = re.sub(
        r"(SurfEval|CurveEval)\((.*?)device='cuda'\)(\.cuda\(\))?",
        r"\1(\2device='cuda')",
        content,
        flags=re.DOTALL
    )

    with open(filepath, 'w') as f:
        f.write(content)

    print(f"Migrated {filepath}")

if __name__ == "__main__":
    for filepath in sys.argv[1:]:
        migrate_file(filepath)
```

Usage:
```bash
python migrate.py examples/*.py
```

## Performance Comparison

| Aspect | v1.x (C++/CUDA) | v2.0 (Pure PyTorch) |
|--------|-----------------|---------------------|
| Installation | Complex | Simple |
| Portability | Limited | Excellent |
| Performance | Fast | Comparable |
| Memory Usage | Lower | Slightly Higher |
| Debugging | Difficult | Easy |
| Gradient Computation | Custom | Native PyTorch |

## Troubleshooting

### Issue: "TypeError: __init__() got unexpected keyword argument 'method'"

**Solution:** Remove `method='tc'` parameter from your initialization.

### Issue: "TypeError: __init__() got unexpected keyword argument 'dvc'"

**Solution:** Replace `dvc='cuda'` with `device='cuda'`.

### Issue: "RuntimeError: Expected all tensors to be on the same device"

**Solution:** Ensure control points are on the same device as the evaluator:
```python
surf_eval = SurfEval(..., device='cuda')
ctrl_pts = ctrl_pts.to('cuda')
```

### Issue: Performance degradation

**Solution:** Ensure you're using GPU acceleration:
```python
# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
surf_eval = SurfEval(..., device=device)
```

## Benefits of Migration

1. **Easier Installation**: No C++ compiler or CUDA toolkit required
2. **Better Portability**: Works on any device PyTorch supports
3. **Cleaner Code**: Using einops makes tensor operations more readable
4. **Easier Debugging**: Pure Python makes debugging much simpler
5. **Better Integration**: Works seamlessly with PyTorch ecosystem
6. **Maintainability**: Easier to extend and modify

## Need Help?

If you encounter issues during migration:
1. Check this guide for common patterns
2. Open an issue on GitHub with your use case
3. See the `examples/` folder for reference implementations

## Rollback

If you need to rollback to v1.x:
```bash
git checkout v1.x
python setup.py develop
```