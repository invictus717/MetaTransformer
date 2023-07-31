#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "ballquery/ballquery_cuda_kernel.h"
#include "knnquery/knnquery_cuda_kernel.h"
#include "sampling/sampling_cuda_kernel.h"
#include "grouping/grouping_cuda_kernel.h"
#include "interpolation/interpolation_cuda_kernel.h"
#include "aggregation/aggregation_cuda_kernel.h"
#include "subtraction/subtraction_cuda_kernel.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("knnquery_cuda", &knnquery_cuda, "knnquery_cuda");
    m.def("ballquery_cuda", &ballquery_cuda, "ballquery_cuda");
    m.def("furthestsampling_cuda", &furthestsampling_cuda, "furthestsampling_cuda");
    m.def("grouping_forward_cuda", &grouping_forward_cuda, "grouping_forward_cuda");
    m.def("grouping_backward_cuda", &grouping_backward_cuda, "grouping_backward_cuda");
    m.def("interpolation_forward_cuda", &interpolation_forward_cuda, "interpolation_forward_cuda");
    m.def("interpolation_backward_cuda", &interpolation_backward_cuda, "interpolation_backward_cuda");
    m.def("subtraction_forward_cuda", &subtraction_forward_cuda, "subtraction_forward_cuda");
    m.def("subtraction_backward_cuda", &subtraction_backward_cuda, "subtraction_backward_cuda");
    m.def("aggregation_forward_cuda", &aggregation_forward_cuda, "aggregation_forward_cuda");
    m.def("aggregation_backward_cuda", &aggregation_backward_cuda, "aggregation_backward_cuda");
}
