"""Pytest configuration and fixtures for NURBSDiff tests."""

import pytest
import torch


@pytest.fixture
def device():
    """Return available device (cuda if available, else cpu)."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def cpu_device():
    """Force CPU device for specific tests."""
    return 'cpu'


@pytest.fixture(params=['cpu', 'cuda'])
def all_devices(request):
    """Parametrize tests across all available devices."""
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    return request.param


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    return 42
