// Copyright (c) 2020,21-22 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//    http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <torch/extension.h>

#include "./ops/mesh_intersection.h"

namespace kaolin {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::module ops = m.def_submodule("ops");
  ops.def("unbatched_mesh_intersection_cuda", &unbatched_mesh_intersection_cuda);
}

}  // namespace kaolin
