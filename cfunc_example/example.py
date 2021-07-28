#!/usr/bin/env python3

"""
Demonstration of passing a :func:`numba.cfunc` through pybind11.

This includes use of a `user_data` pointer.
"""

import ctypes

import numba
import numpy as np
import scipy
from numba import types

from cfunc_example import consumer  # type: ignore


@numba.cfunc(types.intc(types.intc, types.voidptr))
def square(x, user_data):
    """Square `x`, and add `x` to `user_data` (a pointer to intc)."""
    data = numba.carray(user_data, 1, np.intc)
    data[0] += x
    return x * x


def main():  # noqa: D103
    # Allocate an accumulator which will store the sum of all values passed to
    # `square`.
    user_data = ctypes.c_int(0)
    # Wrap `square` into a LowLevelCallable, passing `user_data` as the context.
    callback = scipy.LowLevelCallable(
        square.ctypes, user_data=ctypes.cast(ctypes.pointer(user_data), ctypes.c_void_p)
    )
    # Run sum_func, printing the sum of the squares of the first 10 integers.
    print(consumer.sum_func(10, callback))
    # Print the sum of the integers which was computed on the side.
    print(user_data.value)


if __name__ == "__main__":
    main()
