/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <sstream>
#include <string>

#include "cagra_serialize.cuh"
#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/device_resources.hpp>

namespace cuvs::neighbors::cagra {

#define CUVS_INST_CAGRA_SERIALIZE(DTYPE)                                              \
  void serialize_file(raft::resources const& handle,                                  \
                      const std::string& filename,                                    \
                      const cuvs::neighbors::cagra::index<DTYPE, uint32_t>& index,    \
                      bool include_dataset)                                           \
  {                                                                                   \
    cuvs::neighbors::cagra::serialize_file(handle, filename, index, include_dataset); \
  };                                                                                  \
                                                                                      \
  void deserialize_file(raft::resources const& handle,                                \
                        const std::string& filename,                                  \
                        cuvs::neighbors::cagra::index<DTYPE, uint32_t>* index)        \
  {                                                                                   \
    cuvs::neighbors::cagra::deserialize_file(handle, filename, index);                \
  };                                                                                  \
  void serialize(raft::resources const& handle,                                       \
                 std::string& str,                                                    \
                 const cuvs::neighbors::cagra::index<DTYPE, uint32_t>& index,         \
                 bool include_dataset)                                                \
  {                                                                                   \
    cuvs::neighbors::cagra::serialize(handle, str, index, include_dataset);           \
  }                                                                                   \
                                                                                      \
  void deserialize(raft::resources const& handle,                                     \
                   const std::string& str,                                            \
                   cuvs::neighbors::cagra::index<DTYPE, uint32_t>* index)             \
  {                                                                                   \
    cuvs::neighbors::cagra::deserialize(handle, str, index);                          \
  }

CUVS_INST_CAGRA_SERIALIZE(int8_t);

#undef CUVS_INST_CAGRA_SERIALIZE
}  // namespace cuvs::neighbors::cagra
