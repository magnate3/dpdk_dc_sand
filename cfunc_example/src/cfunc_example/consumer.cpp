/* Declare a functor that takes a function, calls it with integers in the range
 * 1..N and returns the sum of the results. The callback must be given as a
 * tuple whose first element is a PyCapsule. The capsule must
 *
 * - contain a function pointer;
 * - have a name of *exactly* "int (int, void *)";
 * - have a (possibly NULL) context, which is passed as the second argument to
 *   the function pointer.
 *
 * The easiest way to satisfy these requirements is with
 * scipy.LowLevelCallable.
 */

#include <iostream>
#include <pybind11/pybind11.h>

namespace py = pybind11;

static long long sum_func(int N, py::tuple callback)
{
    py::capsule capsule = callback[0];
    if (std::strcmp(capsule.name(), "int (int, void *)") != 0)
        throw std::invalid_argument("Callback has wrong signature");
    // Retrieve the function pointer
    auto fptr = reinterpret_cast<int (*)(int, void *)>(capsule.get_pointer());
    // Retrieve the context
    void *user_data = PyCapsule_GetContext(capsule.ptr());
    // Check for errors from PyCapsule_GetContext (it's raw Python API not
    // pybind11, so doesn't throw on its own).
    if (PyErr_Occurred())
        throw py::error_already_set();
    // Compute the sum.
    long long ans = 0;
    for (int i = 1; i <= N; i++)
        ans += fptr(i, user_data);
    return ans;
}

PYBIND11_MODULE(consumer, m)
{
    m.def("sum_func", sum_func);
}
