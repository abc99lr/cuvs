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

#include "cagra.cuh"
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/cagra_streaming_build.hpp>
#include <iostream>
#include <raft/common/nvtx.hpp>

#include "./detail/cagra/cagra_build.cuh"
#include "./ivf_pq/ivf_pq_build.cuh"
#include "./ivf_pq/ivf_pq_search.cuh"

namespace cuvs::neighbors {

template <typename T, typename IdxT>
cagra_streaming_build<T, IdxT>::cagra_streaming_build(
  raft::resources const& res,
  int64_t num_vectors,
  int64_t dims,
  int64_t num_sample_batch,
  int64_t max_batch_size,
  cuvs::neighbors::cagra::index_params const& params,
  bool ivf_pq_build_using_device_buffer)
  : num_vectors_(num_vectors),
    dims_(dims),
    num_sample_batch_(num_sample_batch),
    max_batch_size_(max_batch_size),
    knn_graph_(res),
    ivf_pq_build_dataset_dev_(res),
    ivf_pq_build_dataset_host_(res),
    ivf_pq_index_(res),
    extend_indices_(res),
    distances_(res),
    neighbors_(res),
    neighbors_host_(res),
    queries_host_(res),
    ivf_index_built_(false)
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope("cagra_streaming_build");
  cagra_params_ = params;

  intermediate_degree_ = cagra_params_.intermediate_graph_degree;
  graph_degree_        = cagra_params_.graph_degree;

  // TODO: we are suppose to check if the intermediate_graph_degree is larger than dataset size.
  if (intermediate_degree_ < graph_degree_) {
    RAFT_LOG_WARN(
      "Graph degree (%lu) cannot be larger than intermediate graph degree (%lu), reducing "
      "graph_degree.",
      graph_degree_,
      intermediate_degree_);
    graph_degree_ = intermediate_degree_;
  }

  if (cagra_params_.attach_dataset_on_build == true) {
    RAFT_LOG_WARN(
      "CAGRA streaming build does not support attaching dataset on build. Forcing to not attach "
      "dataset.");
    cagra_params_.attach_dataset_on_build = false;
  }

  extend_indices_                   = raft::make_host_vector<int64_t, int64_t>(max_batch_size);
  ivf_pq_build_using_device_buffer_ = ivf_pq_build_using_device_buffer;
  auto knn_build_params             = cagra_params_.graph_build_params;
  if (ivf_pq_build_using_device_buffer_) {
    ivf_pq_build_dataset_dev_ =
      raft::make_device_matrix<T, int64_t>(res, num_sample_batch_ * max_batch_size_, dims_);
    if (!std::holds_alternative<cagra::graph_build_params::ivf_pq_params>(knn_build_params)) {
      RAFT_LOG_INFO(
        "Build parameter is not defined or is not defined as IVF-PQ build. Forcing using IVF-PQ "
        "parameters");
      knn_build_params = cagra::graph_build_params::ivf_pq_params(
        ivf_pq_build_dataset_dev_.extents(), cagra_params_.metric);
    }
  } else {
    ivf_pq_build_dataset_host_ =
      raft::make_host_matrix<T, int64_t>(num_sample_batch_ * max_batch_size_, dims_);
    if (!std::holds_alternative<cagra::graph_build_params::ivf_pq_params>(knn_build_params)) {
      RAFT_LOG_INFO(
        "Build parameter is not defined or is not defined as IVF-PQ build. Forcing using IVF-PQ "
        "parameters");
      knn_build_params = cagra::graph_build_params::ivf_pq_params(
        ivf_pq_build_dataset_host_.extents(), cagra_params_.metric);
    }
  }
  // Allocate KNN graph.
  knn_graph_.emplace(raft::make_host_matrix<IdxT, int64_t>(num_vectors_, intermediate_degree_));

  ivf_pq_params_ =
    std::get<cuvs::neighbors::cagra::graph_build_params::ivf_pq_params>(knn_build_params);
  if (ivf_pq_params_.refinement_rate != 1) {
    RAFT_LOG_WARN(
      "CAGRA streaming build does not support refinement for IVF-PQ KNN construction. Forcing to "
      "not using refinement.");
    ivf_pq_params_.refinement_rate = 1;
  }
  if (ivf_pq_params_.build_params.kmeans_trainset_fraction != 1.0) {
    RAFT_LOG_WARN(
      "CAGRA streaming build disables user's control on kmeans_trainset_fraction. Instead all data "
      "points in sampled chunks are used for IVF-PQ KNN construction.");
    ivf_pq_params_.build_params.kmeans_trainset_fraction = 1.0;
  }
  if (ivf_pq_params_.build_params.add_data_on_build) {
    RAFT_LOG_WARN(
      "CAGRA streaming build requires disabling adding data for IVF-PQ KNN construction. Forcing "
      "to not add data to IVF-PQ index.");
    ivf_pq_params_.build_params.add_data_on_build = false;
  }

  top_k_     = intermediate_degree_ + 1;
  gpu_top_k_ = top_k_;  // refinement cannot be not enabled, so set gpu_top_k_ = top_k_.
  // Use the same maximum batch size as the ivf_pq::search to avoid allocating more than needed.
  using cuvs::neighbors::ivf_pq::detail::kMaxQueries;
  kMaxQueries_ = kMaxQueries;

  // Heuristic: the build_knn_graph code should use only a fraction of the workspace memory; the
  // rest should be used by the ivf_pq::search. Here we say that the workspace size should be a good
  // multiple of what is required for the I/O batching below.
  constexpr size_t kMinWorkspaceRatio = 5;
  auto desired_workspace_size         = kMaxQueries_ * kMinWorkspaceRatio *
                                (sizeof(T) * dims_               // queries (dataset batch)
                                 + sizeof(float) * gpu_top_k_    // distances_
                                 + sizeof(int64_t) * gpu_top_k_  // neighbors
                                );

  // If the workspace is smaller than desired, put the I/O buffers into the large workspace.
  rmm::device_async_resource_ref workspace_mr =
    desired_workspace_size <= raft::resource::get_workspace_free_bytes(res)
      ? raft::resource::get_workspace_resource(res)
      : raft::resource::get_large_workspace_resource(res);

  RAFT_LOG_DEBUG(
    "IVF-PQ search node_degree: %d, top_k_: %d,  gpu_top_k_: %d,  max_batch_size:: %d, n_probes: "
    "%u",
    node_degree,
    top_k_,
    gpu_top_k_,
    kMaxQueries,
    pq.search_params.n_probes);

  // Temporary buffers for IVF-PQ search.
  distances_ = raft::make_device_mdarray<float>(
    res, workspace_mr, raft::make_extents<int64_t>(kMaxQueries_, gpu_top_k_));
  neighbors_ = raft::make_device_mdarray<int64_t>(
    res, workspace_mr, raft::make_extents<int64_t>(kMaxQueries_, gpu_top_k_));
  neighbors_host_ = raft::make_host_matrix<int64_t, int64_t>(kMaxQueries_, gpu_top_k_);
  queries_host_   = raft::make_host_matrix<T, int64_t>(kMaxQueries_, dims_);

  sample_batch_processed_ = 0;
  curr_sample_data_size_  = 0;
  curr_graph_data_size_   = 0;
  knn_graph_index_        = 0;
}

// This helper function was borrowed from CAGRA::build implementation. Code related to refinement
// was removed for simplicity.
template <typename T, typename IdxT>
void cagra_streaming_build<T, IdxT>::ivf_pq_search_helper(
  raft::resources const& res, raft::host_matrix_view<const T, int64_t> dataset_view)
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope("ivf_pq_search_helper");
  auto dataset = raft::host_mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major>(
    dataset_view.data_handle(), dataset_view.extent(0), dataset_view.extent(1));

  const auto num_queries = dataset.extent(0);

  // Copied directly from CAGRA build implementation.
  // TODO(tfeher): batched search with multiple GPUs
  std::size_t num_self_included = 0;
  const auto start_clock        = std::chrono::system_clock::now();

  cuvs::spatial::knn::detail::utils::batch_load_iterator<T> vec_batches(
    dataset.data_handle(),
    dataset.extent(0),
    dataset.extent(1),
    static_cast<int64_t>(kMaxQueries_),
    raft::resource::get_cuda_stream(res),
    raft::resource::get_workspace_resource(res));

  size_t next_report_offset = 0;
  size_t d_report_offset    = dataset.extent(0) / 100;  // Report progress in 1% steps.

  size_t previous_batch_size   = 0;
  size_t previous_batch_offset = 0;

  for (const auto& batch : vec_batches) {
    // Map int64_t to uint32_t because ivf_pq requires the latter.
    // TODO(tfeher): remove this mapping once ivf_pq accepts mdspan with int64_t index type
    auto queries_view = raft::make_device_matrix_view<const T, uint32_t>(
      batch.data(), batch.size(), batch.row_width());
    auto neighbors_view = raft::make_device_matrix_view<int64_t, uint32_t>(
      neighbors_.data_handle(), batch.size(), neighbors_.extent(1));
    auto distances_view = raft::make_device_matrix_view<float, uint32_t>(
      distances_.data_handle(), batch.size(), distances_.extent(1));
    cuvs::neighbors::ivf_pq::search(res,
                                    ivf_pq_params_.search_params,
                                    ivf_pq_index_,
                                    queries_view,
                                    neighbors_view,
                                    distances_view);

    auto curr_knn_graph_view = raft::make_host_matrix_view<IdxT, int64_t>(
      knn_graph_->data_handle(), knn_graph_->extent(0), knn_graph_->extent(1));

    // process previous batch async on host
    // NOTE: the async path also covers disabled refinement (top_k_ == gpu_top_k__)
    if (previous_batch_size > 0) {
      cuvs::neighbors::cagra::detail::write_to_graph(curr_knn_graph_view,
                                                     neighbors_host_.view(),
                                                     num_self_included,
                                                     previous_batch_size,
                                                     previous_batch_offset + knn_graph_index_);
    }

    // copy next batch to host
    raft::copy(neighbors_host_.data_handle(),
               neighbors_.data_handle(),
               neighbors_view.size(),
               raft::resource::get_cuda_stream(res));

    previous_batch_size   = batch.size();
    previous_batch_offset = batch.offset();

    // we need to ensure the copy operations are done prior using the host data
    raft::resource::sync_stream(res);

    // process last batch
    if (previous_batch_offset + previous_batch_size == (size_t)num_queries) {
      cuvs::neighbors::cagra::detail::write_to_graph(curr_knn_graph_view,
                                                     neighbors_host_.view(),
                                                     num_self_included,
                                                     previous_batch_size,
                                                     previous_batch_offset + knn_graph_index_);
    }

    size_t num_queries_done = batch.offset() + batch.size();
    const auto end_clock    = std::chrono::system_clock::now();
    if (batch.offset() > next_report_offset) {
      next_report_offset += d_report_offset;
      const auto time =
        std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() *
        1e-6;
      const auto throughput = num_queries_done / time;

      RAFT_LOG_DEBUG(
        "# Search %12lu / %12lu (%3.2f %%), %e queries/sec, %.2f minutes ETA, self included = "
        "%3.2f %%    \r",
        num_queries_done,
        dataset.extent(0),
        num_queries_done / static_cast<double>(dataset.extent(0)) * 100,
        throughput,
        (num_queries - num_queries_done) / throughput / 60,
        static_cast<double>(num_self_included) / num_queries_done * 100.);
    }
  }
  RAFT_LOG_DEBUG("# Finished building kNN graph for current chunk");
}

template <typename T, typename IdxT>
void cagra_streaming_build<T, IdxT>::process_batch(
  raft::resources const& res,
  cuvs::neighbors::cagra::index_params const& params,
  raft::host_matrix_view<const T, int64_t, raft::row_major> dataset,
  bool sample_batch,
  bool add_sample_batch_to_graph)
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope("process_batch");
  int64_t batch_size  = dataset.extent(0);
  int64_t batch_dim   = dataset.extent(1);
  const T* batch_data = dataset.data_handle();

  RAFT_EXPECTS(batch_dim == dims_,
               "Dimensionality of the batch data should be the same as provided in the setup call");
  RAFT_EXPECTS(batch_size <= max_batch_size_,
               "The size for the dataset should be smaller than batch size");

  if (sample_batch) {
    // Copy the current batch to the staging buffer (either on device or on host) for IVF-PQ index
    // build.
    if (sample_batch_processed_ < num_sample_batch_) {
      if (ivf_pq_build_using_device_buffer_) {
        raft::copy(ivf_pq_build_dataset_dev_.data_handle() + curr_sample_data_size_ * dims_,
                   batch_data,
                   batch_size * dims_,
                   raft::resource::get_cuda_stream(res));
      } else {
        raft::copy(ivf_pq_build_dataset_host_.data_handle() + curr_sample_data_size_ * dims_,
                   batch_data,
                   batch_size * dims_,
                   raft::resource::get_cuda_stream(res));
      }
      curr_sample_data_size_ += batch_size;
    }
    sample_batch_processed_++;
    // If the batch is going to be added to the graph, store the pointer to access the batch later.
    if (add_sample_batch_to_graph) {
      graph_data_batch_ptrs_.push_back(batch_data);
      graph_data_batch_sizes_.push_back(batch_size);
    }
    // All sampled batches are received, start to build IVF-PQ index.
    if (sample_batch_processed_ == num_sample_batch_ && !ivf_index_built_) {
      // Build index using ivf_pq_build_dataset.
      if (ivf_pq_build_using_device_buffer_) {
        const std::string model_name = [&]() {
          char model_name[1024];
          sprintf(model_name,
                  "%s-%lux%lu.cluster_%u.pq_%u.%ubit.itr_%u.metric_%u.pqcenter_%u",
                  "IVF-PQ",
                  static_cast<size_t>(ivf_pq_build_dataset_dev_.extent(0)),
                  static_cast<size_t>(ivf_pq_build_dataset_dev_.extent(1)),
                  ivf_pq_params_.build_params.n_lists,
                  ivf_pq_params_.build_params.pq_dim,
                  ivf_pq_params_.build_params.pq_bits,
                  ivf_pq_params_.build_params.kmeans_n_iters,
                  ivf_pq_params_.build_params.metric,
                  static_cast<uint32_t>(ivf_pq_params_.build_params.codebook_kind));
          return std::string(model_name);
        }();
        RAFT_LOG_DEBUG("# Building IVF-PQ index %s", model_name.c_str());
        auto ivf_pq_build_mdspan =
          raft::device_mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major>(
            ivf_pq_build_dataset_dev_.data_handle(), curr_sample_data_size_, dims_);
        ivf_pq_index_ = cuvs::neighbors::ivf_pq::detail::build<T, int64_t>(
          res, ivf_pq_params_.build_params, ivf_pq_build_mdspan);
      } else {
        const std::string model_name = [&]() {
          char model_name[1024];
          sprintf(model_name,
                  "%s-%lux%lu.cluster_%u.pq_%u.%ubit.itr_%u.metric_%u.pqcenter_%u",
                  "IVF-PQ",
                  static_cast<size_t>(ivf_pq_build_dataset_host_.extent(0)),
                  static_cast<size_t>(ivf_pq_build_dataset_host_.extent(1)),
                  ivf_pq_params_.build_params.n_lists,
                  ivf_pq_params_.build_params.pq_dim,
                  ivf_pq_params_.build_params.pq_bits,
                  ivf_pq_params_.build_params.kmeans_n_iters,
                  ivf_pq_params_.build_params.metric,
                  static_cast<uint32_t>(ivf_pq_params_.build_params.codebook_kind));
          return std::string(model_name);
        }();
        RAFT_LOG_DEBUG("# Building IVF-PQ index %s", model_name.c_str());
        auto ivf_pq_build_mdspan =
          raft::host_mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major>(
            ivf_pq_build_dataset_host_.data_handle(), curr_sample_data_size_, dims_);
        ivf_pq_index_ = cuvs::neighbors::ivf_pq::detail::build<T, int64_t>(
          res, ivf_pq_params_.build_params, ivf_pq_build_mdspan);
      }
      ivf_index_built_ = true;

      // Adding previous received batches that are going to be stored in the CAGRA graph to the
      // IVF-PQ index.
      for (size_t i = 0; i < graph_data_batch_ptrs_.size(); i++) {
        // Assign indices in the CAGRA graph for each data point.
        std::iota(extend_indices_.data_handle(),
                  extend_indices_.data_handle() + graph_data_batch_sizes_[i],
                  curr_graph_data_size_);
        curr_graph_data_size_ += graph_data_batch_sizes_[i];
        auto curr_batch_view = raft::make_host_matrix_view<const T, int64_t>(
          graph_data_batch_ptrs_[i], graph_data_batch_sizes_[i], dims_);
        auto extend_indices_view = raft::make_host_vector_view<int64_t, int64_t>(
          extend_indices_.data_handle(), graph_data_batch_sizes_[i]);
        cuvs::neighbors::ivf_pq::extend(res, curr_batch_view, extend_indices_view, &ivf_pq_index_);
      }
    }
  } else {
    // For non-sampled batches, store the pointer for later access and add data to the IVF-PQ index
    // if available.
    graph_data_batch_ptrs_.push_back(batch_data);
    graph_data_batch_sizes_.push_back(batch_size);

    if (!ivf_index_built_) { return; }
    std::iota(extend_indices_.data_handle(),
              extend_indices_.data_handle() + batch_size,
              curr_graph_data_size_);
    auto extend_indices_view =
      raft::make_host_vector_view<int64_t, int64_t>(extend_indices_.data_handle(), batch_size);
    curr_graph_data_size_ += batch_size;
    cuvs::neighbors::ivf_pq::extend(res, dataset, extend_indices_view, &ivf_pq_index_);
  }

  raft::resource::sync_stream(res);
  return;
}

template <typename T, typename IdxT>
auto cagra_streaming_build<T, IdxT>::finalize_graph(
  raft::resources const& res, cuvs::neighbors::cagra::index_params const& params)
  -> cuvs::neighbors::cagra::index<T, IdxT>
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope("finalize_graph");
  // KNN graph construction through IVF-PQ search. Refinement must be disabled since there is no
  // contiguous dataset available.
  for (size_t i = 0; i < graph_data_batch_ptrs_.size(); i++) {
    auto curr_data_batch = raft::make_host_matrix_view<const T, int64_t>(
      graph_data_batch_ptrs_[i], graph_data_batch_sizes_[i], dims_);
    ivf_pq_search_helper(res, curr_data_batch);
    knn_graph_index_ += graph_data_batch_sizes_[i];
  }

  auto cagra_graph = raft::make_host_matrix<IdxT, int64_t>(num_vectors_, graph_degree_);

  RAFT_LOG_INFO("optimizing graph");
  cuvs::neighbors::cagra::detail::optimize<IdxT>(
    res, knn_graph_->view(), cagra_graph.view(), params.guarantee_connectivity);

  // Free intermediate graph before trying to create the index
  knn_graph_.reset();

  RAFT_LOG_INFO("Graph optimized, creating index");

  RAFT_EXPECTS(params.attach_dataset_on_build == false,
               "Dataset cannot be inserted using streaming build");
  RAFT_EXPECTS(params.compression.has_value() == false,
               "Compression is not supported using streaming build");

  cuvs::neighbors::cagra::index<T, IdxT> idx(res);
  // Return CAGRA graph to the user.
  idx.update_graph(res, raft::make_const_mdspan(cagra_graph.view()));
  return idx;
}

template class cagra_streaming_build<float, uint32_t>;
template class cagra_streaming_build<uint8_t, uint32_t>;
}  // namespace cuvs::neighbors