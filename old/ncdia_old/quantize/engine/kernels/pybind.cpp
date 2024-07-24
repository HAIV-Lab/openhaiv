#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "tpack/tpack.h"
#include "functions/funcs.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("tpack", &tpack, "Packs the given tensor into a vector of tensors.");
    m.def("tunpack", &tunpack, "Unpacks the given vector of tensors into a tensor.");
    m.def("linear", &linear, "Linear function.");
    m.def("quantlinear", &quantlinear, "Quantized linear function.");
}
