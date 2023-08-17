
#include <pybind11/pybind11.h>

namespace py = pybind11;

extern void bind_lerp(py::module& m);

PYBIND11_MODULE(cuda_python, m) {   

    m.attr("__version__") = "1.0.0";
  
    py::module cuda_common_interface = m.def_submodule("common", "Cuda accelerated stuff");
    bind_lerp(cuda_common_interface);    
}
