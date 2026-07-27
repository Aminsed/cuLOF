/**
 * pybind11 bindings. Deliberately thin: shape and dtype coercion happens here,
 * everything else is Python (culof/__init__.py) or CUDA (src/*.cu).
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "culof.h"

namespace py = pybind11;

namespace {

// c_style | forcecast makes pybind materialise a C-contiguous float32 array
// whatever the caller passes. Without it, a non-contiguous view such as X[::2]
// keeps its original strides, and reading it as a flat buffer silently returns
// the wrong answer.
using Array = py::array_t<float, py::array::c_style | py::array::forcecast>;

py::array_t<float> lof_py(const Array& points, int k, bool normalize) {
    const auto buf = points.request();
    if (buf.ndim != 2) {
        throw std::invalid_argument(
            "expected a 2-D array of shape (n_samples, n_features)");
    }

    std::vector<float> scores;
    {
        py::gil_scoped_release release;
        scores = culof::lof(static_cast<const float*>(buf.ptr),
                            static_cast<int>(buf.shape[0]),
                            static_cast<int>(buf.shape[1]), k, normalize);
    }

    py::array_t<float> out(static_cast<py::ssize_t>(scores.size()));
    std::copy(scores.begin(), scores.end(), out.mutable_data());
    return out;
}

}  // namespace

PYBIND11_MODULE(_culof, m) {
    m.doc() = "cuLOF native extension";

    m.def("lof", &lof_py, py::arg("points"), py::arg("k"), py::arg("normalize"),
          "Raw LOF scores for a (n_samples, n_features) array.");
    m.def("cuda_available", &culof::cuda_available);
    m.def("device_info", &culof::device_info);

    py::register_exception<culof::CudaError>(m, "CudaError", PyExc_RuntimeError);
}
