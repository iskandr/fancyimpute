import numpy as np

from fancyimpute.dictionary_helpers import (
    dense_matrix_from_pair_dictionary,
    dense_matrix_from_nested_dictionary,
    reverse_lookup_from_nested_dict,
    transpose_nested_dictionary,
)
from nose.tools import eq_


def test_dense_matrix_from_nested_dictionary():
    d = {
        "a": {"b": 10},
        "b": {"c": 20}
    }
    X, rows, columns = dense_matrix_from_nested_dictionary(d)
    eq_(rows, ["a", "b"])
    eq_(columns, ["b", "c"])
    eq_(X[0, 0], 10)
    assert np.isnan(X[0, 1])
    assert np.isnan(X[1, 0])
    eq_(X[1, 1], 20)


def test_dense_matrix_from_nested_dictionary_square():
    d = {
        "a": {"b": 10},
        "b": {"c": 20}
    }
    X, rows, columns = dense_matrix_from_nested_dictionary(d, square_result=True)
    eq_(rows, ["a", "b", "c"])
    eq_(columns, ["a", "b", "c"])
    assert np.isnan(X[0, 0])
    eq_(X[0, 1], 10)
    assert np.isnan(X[0, 2])
    assert np.isnan(X[1, 0])
    assert np.isnan(X[1, 1])
    eq_(X[1, 2], 20)
    assert np.isnan(X[2, 0])
    assert np.isnan(X[2, 1])
    assert np.isnan(X[2, 2])


def test_dense_matrix_from_pair_dictionary():
    d = {
        ("a", "b"): 10,
        ("b", "c"): 20
    }
    X, rows, columns = dense_matrix_from_pair_dictionary(d)
    eq_(rows, ["a", "b"])
    eq_(columns, ["b", "c"])
    eq_(X[0, 0], 10)
    assert np.isnan(X[0, 1])
    assert np.isnan(X[1, 0])
    eq_(X[1, 1], 20)


def test_dense_matrix_from_pair_dictionary_square():
    d = {
        ("a", "b"): 10,
        ("b", "c"): 20
    }
    X, rows, columns = dense_matrix_from_pair_dictionary(d, square_result=True)
    eq_(rows, ["a", "b", "c"])
    eq_(columns, ["a", "b", "c"])
    assert np.isnan(X[0, 0])
    eq_(X[0, 1], 10)
    assert np.isnan(X[0, 2])
    assert np.isnan(X[1, 0])
    assert np.isnan(X[1, 1])
    eq_(X[1, 2], 20)
    assert np.isnan(X[2, 0])
    assert np.isnan(X[2, 1])
    assert np.isnan(X[2, 2])


def test_reverse_lookup_from_nested_dict():
    d = {
        "a": {"b": 10, "c": 20},
        "b": {"c": 5},
        "z": {"c": 100}
    }
    reverse_dict = reverse_lookup_from_nested_dict(d)
    len(reverse_dict.keys()) == 2
    assert "c" in reverse_dict
    eq_(set(reverse_dict["c"]), {("a", 20), ("b", 5), ("z", 100)})
    assert "b" in reverse_dict
    eq_(reverse_dict["b"], [("a", 10)])


def test_transpose_nested_dictionary():
    d = {"a": {"b": 20, "c": 50}, "c": {"q": 500}}
    transposed = transpose_nested_dictionary(d)
    eq_(set(transposed.keys()), {"b", "c", "q"})
    eq_(transposed["q"], {"c": 500})
    eq_(transposed["c"], {"a": 50})
    eq_(transposed["b"], {"a": 20})
