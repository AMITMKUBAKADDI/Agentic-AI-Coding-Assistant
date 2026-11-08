import pytest
from calculator import add, subtract, multiply, divide

def test_add_positive_numbers():
    assert add(2, 3) == 5

def test_add_negative_numbers():
    assert add(-2, -3) == -5

def test_add_mixed_numbers():
    assert add(-2, 3) == 1
    assert add(2, -3) == -1

def test_add_with_zero():
    assert add(5, 0) == 5
    assert add(0, 5) == 5
    assert add(0, 0) == 0

def test_add_float_numbers():
    assert add(2.5, 3.5) == 6.0
    assert add(-1.5, 2.0) == 0.5

def test_subtract_positive_numbers():
    assert subtract(5, 2) == 3

def test_subtract_negative_numbers():
    assert subtract(-5, -2) == -3

def test_subtract_mixed_numbers():
    assert subtract(5, -2) == 7
    assert subtract(-5, 2) == -7

def test_subtract_with_zero():
    assert subtract(5, 0) == 5
    assert subtract(0, 5) == -5
    assert subtract(0, 0) == 0

def test_subtract_float_numbers():
    assert subtract(5.5, 2.5) == 3.0
    assert subtract(2.0, 1.5) == 0.5

def test_subtract_order_regression():
    assert subtract(10, 4) == 6
    assert subtract(4, 10) == -6

def test_multiply_positive_numbers():
    assert multiply(2, 3) == 6

def test_multiply_negative_numbers():
    assert multiply(-2, -3) == 6

def test_multiply_mixed_numbers():
    assert multiply(-2, 3) == -6
    assert multiply(2, -3) == -6

def test_multiply_with_zero():
    assert multiply(5, 0) == 0
    assert multiply(0, 5) == 0
    assert multiply(0, 0) == 0

def test_multiply_with_one():
    assert multiply(5, 1) == 5
    assert multiply(1, 5) == 5

def test_multiply_float_numbers():
    assert multiply(2.5, 2.0) == 5.0
    assert multiply(-1.5, 2.0) == -3.0

def test_divide_positive_numbers():
    assert divide(6, 3) == 2.0

def test_divide_negative_numbers():
    assert divide(-6, -3) == 2.0

def test_divide_mixed_numbers():
    assert divide(-6, 3) == -2.0
    assert divide(6, -3) == -2.0

def test_divide_by_one():
    assert divide(5, 1) == 5.0

def test_divide_float_numbers():
    assert divide(7.5, 2.5) == 3.0
    assert divide(5.0, 2.0) == 2.5

def test_divide_by_zero_regression():
    with pytest.raises(ValueError): # As per project specification, expecting ValueError
        divide(10, 0)