/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#pragma once

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/mdspan_types.hpp>

namespace cuvs::neighbors {
template <typename T, typename IdxT>
class cagra_streaming_build {
 public:
  cagra_streaming_build()  = default;
  ~cagra_streaming_build() = default;

  cagra_streaming_build(raft::resources const& res,
                        int64_t num_vectors,
                        int64_t dims,
                        int64_t num_sample_batch,
                        int64_t max_batch_size,
                        cuvs::neighbors::cagra::index_params const& params,
                        bool ivf_pq_build_using_device_buffer = true);

  void process_batch(raft::resources const& res,
                     cuvs::neighbors::cagra::index_params const& params,
                     raft::host_matrix_view<const T, int64_t, raft::row_major> dataset,
                     bool sample_batch,
                     bool add_sample_batch_to_graph);

  auto finalize_graph(raft::resources const& res,
                      cuvs::neighbors::cagra::index_params const& params)
    -> cuvs::neighbors::cagra::index<T, IdxT>;

 private:
  void ivf_pq_search_helper(raft::resources const& res,
                            raft::host_matrix_view<const T, int64_t> dataset_view);

  int64_t num_vectors_;
  int64_t dims_;
  int64_t num_sample_batch_;
  int64_t max_batch_size_;

  int64_t sample_batch_processed_;
  int64_t curr_sample_data_size_;
  int64_t curr_graph_data_size_;
  int64_t knn_graph_index_;

  int64_t intermediate_degree_;
  int64_t graph_degree_;

  // Parameters to for CAGRA and IVF-PQ.
  cuvs::neighbors::cagra::index_params cagra_params_;

  // Temporary device buffers.
  std::optional<raft::host_matrix<IdxT, int64_t>> knn_graph_;

  std::vector<const T*> graph_data_batch_ptrs_;
  std::vector<int64_t> graph_data_batch_sizes_;
  bool device_buffer_for_ivf_build_;

  // Device and host staging buffers for IVF-PQ index build. Only one is used based on user input.
  raft::device_matrix<T, int64_t, raft::row_major> ivf_pq_build_dataset_dev_;
  raft::host_matrix<T, int64_t, raft::row_major> ivf_pq_build_dataset_host_;
  bool ivf_pq_build_using_device_buffer_;

  raft::host_vector<int64_t, int64_t, raft::row_major> extend_indices_;

  cuvs::neighbors::ivf_pq::index<int64_t> ivf_pq_index_;
  cuvs::neighbors::cagra::graph_build_params::ivf_pq_params ivf_pq_params_;

  int64_t top_k_;
  int64_t gpu_top_k_;

  using extents = std::experimental::extents<int64_t, raft::dynamic_extent, raft::dynamic_extent>;

  raft::device_mdarray<float, extents, raft::row_major> distances_;
  raft::device_mdarray<int64_t, extents, raft::row_major> neighbors_;
  raft::host_matrix<int64_t, int64_t, raft::row_major> neighbors_host_;
  raft::host_matrix<T, int64_t, raft::row_major> queries_host_;

  uint32_t kMaxQueries_;
  bool ivf_index_built_;
};

extern template class cagra_streaming_build<float, uint32_t>;
extern template class cagra_streaming_build<uint8_t, uint32_t>;
}  // namespace cuvs::neighbors