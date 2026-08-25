from addons import *
import pytest

@uusing("einsums")
@ctest_labeler("quick;x2c")
@pytest.mark.skip("Temporarily disabling SOC X2C regression tests for testing.")
def test_sox2c1():
    ctest_runner(__file__)

