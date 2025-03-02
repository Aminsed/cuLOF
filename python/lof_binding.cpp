#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "cuda_lof.h"

namespace py = pybind11;

// Helper to convert NumPy arrays to our internal format
std::vector<float> numpy_to_vector(py::array_t<float> array) {
    py::buffer_info buf = array.request();
    
    if (buf.ndim != 2) {
        throw std::runtime_error("Input array must be 2-dimensional");
    }
    
    float* ptr = static_cast<float*>(buf.ptr);
    int n_points = buf.shape[0];
    int n_dims = buf.shape[1];
    
    std::vector<float> result(n_points * n_dims);
    for (int i = 0; i < n_points; i++) {
        for (int j = 0; j < n_dims; j++) {
            // Handle both C and Fortran contiguous arrays
            if (buf.strides[1] == sizeof(float)) {
                // C-contiguous
                result[i * n_dims + j] = ptr[i * buf.strides[0] / sizeof(float) + j];
            } else {
                // Fortran-contiguous or other
                result[i * n_dims + j] = ptr[i * buf.strides[0] / sizeof(float) + j * buf.strides[1] / sizeof(float)];
            }
        }
    }
    
    return result;
}

// NumPy array (n_points, n_dims) -> LOF -> NumPy array (n_points,)
py::array_t<float> compute_lof_py(py::array_t<float> points, int k = 20, bool normalize = true, float threshold = 1.5f) {
    // Get array info
    py::buffer_info buf = points.request();
    
    if (buf.ndim != 2) {
        throw std::runtime_error("Input array must be 2-dimensional");
    }
    
    // Create LOF object
    LOF lof(k, normalize, threshold);
    
    // Convert input array to our format
    float* ptr = static_cast<float*>(buf.ptr);
    int n_points = buf.shape[0];
    int n_dims = buf.shape[1];
    
    std::vector<float> result;
    
    // Release GIL for the compute-intensive part
    {
        py::gil_scoped_release release;
        
        // Compute LOF scores
        if (buf.strides[1] == sizeof(float)) {
            // C-contiguous array - we can use the direct pointer
            result = lof.fit_predict(ptr, n_points, n_dims);
        } else {
            // Non-contiguous array - need to copy
            std::vector<float> data = numpy_to_vector(points);
            result = lof.fit_predict(data.data(), n_points, n_dims);
        }
    }
    
    // Create output array
    py::array_t<float> output(n_points);
    py::buffer_info out_buf = output.request();
    float* out_ptr = static_cast<float*>(out_buf.ptr);
    
    // Copy results to output array
    for (int i = 0; i < n_points; i++) {
        out_ptr[i] = result[i];
    }
    
    return output;
}

// Get outliers based on threshold
py::array_t<int> get_outliers_py(py::array_t<float> scores, float threshold = 1.5f) {
    py::buffer_info buf = scores.request();
    
    if (buf.ndim != 1) {
        throw std::runtime_error("Scores array must be 1-dimensional");
    }
    
    float* ptr = static_cast<float*>(buf.ptr);
    int n_points = buf.shape[0];
    
    // Create temporary vector
    std::vector<float> scores_vec(n_points);
    for (int i = 0; i < n_points; i++) {
        scores_vec[i] = ptr[i];
    }
    
    // Create LOF object just to use get_outliers
    LOF lof(20, true, threshold);
    std::vector<int> outliers = lof.get_outliers(scores_vec);
    
    // Create output array
    py::array_t<int> output(outliers.size());
    py::buffer_info out_buf = output.request();
    int* out_ptr = static_cast<int*>(out_buf.ptr);
    
    // Copy outlier indices to output array
    for (size_t i = 0; i < outliers.size(); i++) {
        out_ptr[i] = outliers[i];
    }
    
    return output;
}

// Python module definition
PYBIND11_MODULE(_cuda_lof, m) {
    m.doc() = "CUDA-accelerated Local Outlier Factor implementation";
    
    // Add functions
    m.def("compute_lof", &compute_lof_py, py::arg("points"), py::arg("k") = 20, 
          py::arg("normalize") = true, py::arg("threshold") = 1.5f,
          "Compute LOF scores for each point in the dataset");
    
    m.def("get_outliers", &get_outliers_py, py::arg("scores"), py::arg("threshold") = 1.5f,
          "Get indices of outliers based on threshold");
    
    // Add LOF class
    py::class_<LOF>(m, "LOF")
        .def(py::init<int, bool, float, int>(),
             py::arg("k") = 20,
             py::arg("normalize") = true,
             py::arg("threshold") = 1.5f,
             py::arg("min_points") = -1)
        .def("fit_predict", [](LOF& self, py::array_t<float> points) {
            py::buffer_info buf = points.request();
            
            if (buf.ndim != 2) {
                throw std::runtime_error("Input array must be 2-dimensional");
            }
            
            float* ptr = static_cast<float*>(buf.ptr);
            int n_points = buf.shape[0];
            int n_dims = buf.shape[1];
            
            std::vector<float> result;
            
            // Release GIL during computation
            {
                py::gil_scoped_release release;
                
                if (buf.strides[1] == sizeof(float)) {
                    // C-contiguous array
                    result = self.fit_predict(ptr, n_points, n_dims);
                } else {
                    // Non-contiguous array
                    std::vector<float> data = numpy_to_vector(points);
                    result = self.fit_predict(data.data(), n_points, n_dims);
                }
            }
            
            // Create output array
            py::array_t<float> output(n_points);
            py::buffer_info out_buf = output.request();
            float* out_ptr = static_cast<float*>(out_buf.ptr);
            
            // Copy results to output array
            for (int i = 0; i < n_points; i++) {
                out_ptr[i] = result[i];
            }
            
            return output;
        })
        .def("get_outliers", [](LOF& self, py::array_t<float> scores) {
            py::buffer_info buf = scores.request();
            
            if (buf.ndim != 1) {
                throw std::runtime_error("Scores array must be 1-dimensional");
            }
            
            float* ptr = static_cast<float*>(buf.ptr);
            int n_points = buf.shape[0];
            
            // Create temporary vector
            std::vector<float> scores_vec(n_points);
            for (int i = 0; i < n_points; i++) {
                scores_vec[i] = ptr[i];
            }
            
            std::vector<int> outliers = self.get_outliers(scores_vec);
            
            // Create output array
            py::array_t<int> output(outliers.size());
            py::buffer_info out_buf = output.request();
            int* out_ptr = static_cast<int*>(out_buf.ptr);
            
            // Copy outlier indices to output array
            for (size_t i = 0; i < outliers.size(); i++) {
                out_ptr[i] = outliers[i];
            }
            
            return output;
        })
        .def("set_k", &LOF::set_k)
        .def("set_normalize", &LOF::set_normalize)
        .def("set_threshold", &LOF::set_threshold);
    
    // Add version info
    m.attr("__version__") = "0.1.0";
} 