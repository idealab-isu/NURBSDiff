# NURBSDiff Tests

Comprehensive test suite for NURBSDiff, covering all core functionality including utility functions, curve evaluation, and surface evaluation.

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test File
```bash
pytest tests/test_utils.py
pytest tests/test_curve_eval.py
pytest tests/test_surf_eval.py
pytest tests/test_nurbs_eval.py
```

### Run with Coverage
```bash
pytest tests/ --cov=NURBSDiff --cov-report=html
```

### Run with Verbose Output
```bash
pytest tests/ -v
```

### Run Only Fast Tests (Skip CUDA tests)
```bash
pytest tests/ -m "not cuda"
```

## Test Organization

### `test_utils.py`
Tests for utility functions in `NURBSDiff/utils.py`:
- **TestKnotVectorGeneration**: Knot vector generation
- **TestFindSpan**: Finding knot spans
- **TestBasisFunctions**: B-spline basis function computation
- **TestUtilsIntegration**: Integration tests

### `test_curve_eval.py`
Tests for NURBS curve evaluation in `NURBSDiff/curve_eval.py`:
- **TestCurveEvalInit**: Initialization tests
- **TestCurveEvalForward**: Forward pass tests
- **TestCurveEvalGradients**: Gradient computation tests
- **TestCurveEvalAccuracy**: Numerical accuracy tests
- **TestCurveEvalEdgeCases**: Edge cases
- **TestCurveEvalConsistency**: Consistency across devices

### `test_surf_eval.py`
Tests for NURBS surface evaluation in `NURBSDiff/surf_eval.py`:
- **TestSurfEvalInit**: Initialization tests
- **TestSurfEvalForward**: Forward pass tests
- **TestSurfEvalGradients**: Gradient computation tests
- **TestSurfEvalAccuracy**: Numerical accuracy tests
- **TestSurfEvalEdgeCases**: Edge cases
- **TestSurfEvalConsistency**: Consistency across devices

### `test_nurbs_eval.py`
Tests for NURBS evaluation with learnable knots in `NURBSDiff/nurbs_eval.py`:
- **TestNURBSEvalInit**: Initialization tests
- **TestNURBSEvalForward**: Forward pass tests
- **TestNURBSEvalGradients**: Gradient computation (control points and knots)
- **TestNURBSEvalRobustness**: Robustness tests
- **TestNURBSEvalEdgeCases**: Edge cases
- **TestNURBSEvalConsistency**: Consistency across devices

## Test Coverage

The test suite covers:
- ✅ Basic functionality
- ✅ Gradient computation (autograd)
- ✅ Numerical accuracy
- ✅ Edge cases
- ✅ CPU and CUDA consistency
- ✅ Different parameter configurations
- ✅ Batch processing
- ✅ Error handling

## Requirements

```bash
pip install pytest pytest-cov
```

## CI/CD

Tests can be run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pytest tests/ -v --cov=NURBSDiff
```

## Writing New Tests

When adding new functionality:

1. Create tests in the appropriate test file
2. Follow the existing test structure
3. Use parametrize for multiple test cases
4. Include tests for both CPU and CUDA
5. Test edge cases and error conditions

Example:
```python
@pytest.mark.parametrize("device", ['cpu', 'cuda'])
def test_new_feature(self, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    # Your test code here
    pass
```
