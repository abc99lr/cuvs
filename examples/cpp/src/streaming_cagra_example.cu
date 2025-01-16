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

#include <cstdint>
#include <algorithm>
#include <vector>
#include <getopt.h>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/random/make_blobs.cuh>

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/cagra_streaming_build.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <nvtx3/nvToolsExt.h>
#include "common.cuh"
#include "data_loader.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

using namespace cuvs::neighbors;

// Helper function to evaluate recall.
double calculate_recall(
  raft::host_matrix_view<const uint32_t, int64_t> groundtruth_host, 
  raft::host_matrix_view<const uint32_t, int64_t> neighbors_host) 
{
  int64_t max_k = groundtruth_host.extent(1);
  int64_t k = neighbors_host.extent(1);
  if (groundtruth_host.extent(0) != neighbors_host.extent(0) || max_k < k) {
      return -1;
  }
  const uint32_t* groundtruth_neighbor_ptr = groundtruth_host.data_handle();
  const uint32_t* neighbor_ptr = neighbors_host.data_handle();

  // Borrowed from https://github.com/rapidsai/cuvs/blob/branch-24.10/cpp/bench/ann/src/common/benchmark.hpp#L342
  std::size_t n_queries = groundtruth_host.extent(0);
  std::size_t match_count = 0;
  std::size_t total_count = n_queries * static_cast<size_t>(k);

  // We go through the groundtruth with same stride as the benchmark loop.
  for (std::size_t i = 0; i < n_queries; i++) {
    for (std::uint32_t j = 0; j < k; j++) {
      auto act_idx = neighbor_ptr[i * k + j];
      for (std::uint32_t l = 0; l < k; l++) {
        auto exp_idx = groundtruth_neighbor_ptr[i * max_k + l];
        if (act_idx == exp_idx) {
          match_count++;
          break;
        }
      }
    }
  }
  return static_cast<double>(match_count) / static_cast<double>(total_count);
} 

// Helper function for CAGRA search.
void search_helper(
  raft::device_resources const& dev_resources, 
  const std::list<size_t>& itopk_list,
  std::vector<float>& recall_res,
  const cuvs::neighbors::cagra::index<float, uint32_t>& index,
  raft::host_matrix_view<const float, int64_t> query_host,
  raft::host_matrix_view<const uint32_t, int64_t> groundtruth_host, 
  const int64_t topk,
  const bool verbose)
{
  // Creating device query set, neighbor array, distance array, as well as host neighbor array.
  int64_t n_queries = query_host.extent(0);
  int64_t n_dim = query_host.extent(1);
  auto query_device = raft::make_device_matrix<float>(dev_resources, n_queries, n_dim);
  auto neighbors_device = raft::make_device_matrix<uint32_t>(dev_resources, n_queries, topk);
  auto distances_device = raft::make_device_matrix<float>(dev_resources, n_queries, topk);
  auto neighbors_host = raft::make_host_matrix<uint32_t>(n_queries, topk);

  // Copying query from host matrix to device matrix. 
  raft::copy(query_device.data_handle(), query_host.data_handle(), query_host.size(), raft::resource::get_cuda_stream(dev_resources));  

  cagra::search_params search_params;
  search_params.max_iterations = 512;
  search_params.search_width = 8;
  if (verbose) {
    std::cout << "itopk,recall" << std::endl;
  }
  for (auto it = itopk_list.begin(); it != itopk_list.end(); ++it) {
    search_params.itopk_size = *it; 
    cagra::search(dev_resources, search_params, index, query_device.view(), neighbors_device.view(), distances_device.view());
    // Copying neighbors back to host for recall computation. 
    raft::copy(neighbors_host.data_handle(), neighbors_device.data_handle(), neighbors_device.size(), raft::resource::get_cuda_stream(dev_resources));  
    double recall = calculate_recall(groundtruth_host, neighbors_host.view());
    if (verbose) {
      std::cout << search_params.itopk_size << "," << recall << std::endl;  
    }
    recall_res.push_back(recall);
  }
}

// Normal CAGRA index build.
void baseline_build_search(
  raft::device_resources const& dev_resources, 
  const std::list<size_t>& itopk_list,
  std::vector<float>& recall_res,
  raft::host_matrix_view<const float, int64_t> dataset, 
  raft::host_matrix_view<const float, int64_t> query, 
  raft::host_matrix_view<const uint32_t, int64_t> groundtruth, 
  cagra::index_params& index_params, 
  const int64_t topk, 
  const bool verbose) 
{
  std::cout << "Building baseline CAGRA index" << std::endl;
  auto pq_params = cagra::graph_build_params::ivf_pq_params(dataset.extents(), cuvs::distance::DistanceType::L2Expanded);
  // Disable refinement and using IVF-PQ KNN construction to match streaming CAGRA build.
  pq_params.refinement_rate = 1;
  index_params.attach_dataset_on_build = false;
  index_params.graph_build_params = pq_params;
  auto start = std::chrono::high_resolution_clock::now();
  auto index = cagra::build(dev_resources, index_params, dataset);
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Build elapsed time: " << elapsed.count() << " milliseconds\n";

  index.update_dataset(dev_resources, raft::make_const_mdspan(dataset));
  if (verbose) {
    std::cout << "CAGRA index has " << index.size() << " vectors" << std::endl;
    std::cout << "CAGRA graph has degree " << index.graph_degree() << ", graph size ["
              << index.graph().extent(0) << ", " << index.graph().extent(1) << "]" << std::endl;
    std::cout << "Searching baseline CAGRA index" << std::endl;
  }
  search_helper(dev_resources, itopk_list, recall_res, index, query, groundtruth, topk, verbose);
}

// Streaming CAGRA index build.
void streaming_build_search(
  raft::device_resources const& dev_resources, 
  const std::list<size_t>& itopk_list,
  std::vector<float>& recall_res,
  raft::host_matrix_view<const float, int64_t> dataset, 
  raft::host_matrix_view<const float, int64_t> query, 
  raft::host_matrix_view<const uint32_t, int64_t> groundtruth, 
  cagra::index_params& index_params, 
  const int64_t topk, 
  const int64_t streaming_batch_size, 
  const float kmeans_percentage, 
  const bool train_kmeans_with_beginning_batches,
  const bool verbose)
{
  std::cout << "Streaming building CAGRA index" << std::endl;
  int64_t num_dim = dataset.extent(1);
  int64_t num_batch = ceil(1.0 * dataset.extent(0) / streaming_batch_size);
  int64_t num_sampled_batch = ceil(kmeans_percentage * num_batch);
  // Only CAGRA graph output is needed.
  index_params.attach_dataset_on_build = false;
  // IVF-PQ KNN construction is required for streaming build.
  auto pq_params = cagra::graph_build_params::ivf_pq_params(dataset.extents(), cuvs::distance::DistanceType::L2Expanded);
  // Refinement rate=1 is required for streaming build. 
  pq_params.refinement_rate = 1;
  index_params.graph_build_params = pq_params;
  // Decide which batches to use for IVF-PQ index building. Either using beginning batches or random batches.
  std::set<int> selected_batches;
  if (train_kmeans_with_beginning_batches) {
    for (int i = 0; i < num_sampled_batch; ++i) {
      selected_batches.insert(i);
    }
  } else {
    std::random_device rd; 
    std::mt19937 g(12345);
    std::uniform_int_distribution<> dist(0, num_batch - 1);
    while (selected_batches.size() < num_sampled_batch) {
      int batch_id = dist(g);
      selected_batches.insert(batch_id);
    }
  }
  
  if (verbose) {
    std::cout << "Batch size for streaming build: " << streaming_batch_size << std::endl; 
    std::cout << "Number of batches: " << num_batch << std::endl; 
    std::cout << "Batch ids used for building IVF-PQ: "; 
    for (const auto& id : selected_batches) {
      std::cout << id << " ";
    }
    std::cout << std::endl;
  }

  auto start = std::chrono::high_resolution_clock::now();
  nvtxRangePush("StreamingCAGRA_construction");
  cuvs::neighbors::cagra_streaming_build<float, uint32_t> test(dev_resources, dataset.extent(0), num_dim, 
    num_sampled_batch, streaming_batch_size, index_params, false);
  nvtxRangePop();

  for (int i = 0; i < num_batch; i++) {
    nvtxRangePush("StreamingCAGRA_process_batch");
    int64_t curr_batch_size = std::min(streaming_batch_size, (int64_t)dataset.extent(0) - i * streaming_batch_size);
    auto curr_dataset = raft::make_host_matrix_view<const float, int64_t>(dataset.data_handle() + i * streaming_batch_size * num_dim, curr_batch_size, num_dim);
    if (selected_batches.find(i) != selected_batches.end()) {
      // Sampled batches for IVF-PQ index building.
      test.process_batch(dev_resources, index_params, curr_dataset, true, true);
    } else {
      // Non-sampled batches.
      test.process_batch(dev_resources, index_params, curr_dataset, false, false);
    }
    nvtxRangePop();
  }
  auto end_add_batch = std::chrono::high_resolution_clock::now();
  // Finialize the graph building process.
  nvtxRangePush("StreamingCAGRA_finalize_graph");
  auto index = test.finalize_graph(dev_resources, index_params);
  nvtxRangePop();
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Build elapsed time: " << elapsed.count() << " milliseconds\n";
  auto elapsed_add_batch = std::chrono::duration_cast<std::chrono::milliseconds>(end_add_batch - start);
  std::cout << "Adding batch elapsed time: " << elapsed_add_batch.count() << " milliseconds\n";

  // Update dataset is required before search. However, if the user only needs CAGRA graph, it's already returned in index. No need to update dataset.  
  index.update_dataset(dev_resources, raft::make_const_mdspan(dataset));
  if (verbose) {
    std::cout << "CAGRA index has " << index.size() << " vectors" << std::endl;
    std::cout << "CAGRA graph has degree " << index.graph_degree() << ", graph size ["
              << index.graph().extent(0) << ", " << index.graph().extent(1) << "]" << std::endl;
    std::cout << "Searching baseline CAGRA index" << std::endl;
  }
  search_helper(dev_resources, itopk_list, recall_res, index, query, groundtruth, topk, verbose);
}

void print_helper(const struct option* longOptions) {
  const char* descriptions[] = {
    "Path to dataset file (string)",
    "Path to query file (string)",
    "Path to groundtruth file (string)",
    "CAGRA graph degree (int)",
    "Streaming build batch size (int)",
    "Top-K (int)",
    "Streaming build k-means training percentage (float)",
    "Enable data permutation for streaming build (bool)",
    "Skip baseline CAGRA build (bool)",
    "Streaming build train kmeans with beginning batches (bool)",
    "Allocation with host pinned memory (bool)",
    "Verbose (bool)",
    "Show this help message"
  };

  std::cout << "Usage: STREAMING_CAGRA_EXAMPLE [OPTIONS]\n";
  std::cout << "Options:\n";

  for (int i = 0; longOptions[i].name != 0; ++i) {
    std::cout << "  -" << (char)longOptions[i].val << ", --" << longOptions[i].name;
    if (longOptions[i].has_arg == required_argument) {
        std::cout << " <arg>";
    }
    std::cout << "    " << descriptions[i] << std::endl;
  }
}

int main(int argc, char* argv[]) 
{
  std::string dataset_fname = "../data/sift-128-euclidean/base.fbin";
  std::string query_fname = "../data/sift-128-euclidean/query.fbin";
  std::string groundtruth_fname = "../data/sift-128-euclidean/groundtruth.neighbors.ibin";
  int64_t topk = 10;
  int64_t graph_degree = 32;
  int64_t streaming_batch_size = 100'000;
  float kmeans_percentage = 0.2;
  bool skip_baseline = false;
  bool data_permutation = false;
  bool train_kmeans_with_beginning_batches = false;
  bool verbose = false;
  bool allocate_pinned = false;

  struct option longOptions[] = {
    {"dataset", required_argument, 0, 'd'},
    {"query", required_argument, 0, 'q'},
    {"groundtruth", required_argument, 0, 't'},
    {"graph_degree", required_argument, 0, 'g'},
    {"streaming_batch_size", required_argument, 0, 'b'},
    {"topk", required_argument, 0, 'k'},
    {"kmeans_percentage", required_argument, 0, 'p'},
    {"data_permutation", no_argument, 0, 'P'},
    {"skip_baseline", no_argument, 0, 'S'},
    {"train_kmeans_with_beginning_batches", no_argument, 0, 'B'},
    {"allocate_pinned", no_argument, 0, 'I'},
    {"verbose", no_argument, 0, 'V'},
    {"help", no_argument, 0, 'h'},
    {0, 0, 0, 0} // Terminating zeroed out element
  };

  int optionIndex = 0;
  int opt;
  while ((opt = getopt_long(argc, argv, "d:q:t:g:b:k:p:PSBIVh", longOptions, &optionIndex)) != -1) {
    switch (opt) {
      case 'd':
        dataset_fname = optarg;
        break;
      case 'q':
        query_fname = optarg;
        break;
      case 't':
        groundtruth_fname = optarg;
        break;
      case 'g':
        graph_degree = std::atol(optarg);
        break;
      case 'b':
        streaming_batch_size = std::atol(optarg);
        break;
      case 'k':
        topk = std::atol(optarg);
        break;
      case 'p':
        kmeans_percentage = std::atof(optarg);
        break;
      case 'P':
        data_permutation = true;
        break;
      case 'S':
        skip_baseline = true;
        break;
      case 'B':
        train_kmeans_with_beginning_batches = true;
        break;
      case 'I':
        allocate_pinned = true;
        break;
      case 'V':
        verbose = true; 
        break; 
      case 'h':
        print_helper(longOptions);
        return 0; 
      default:
        std::cerr << "Invalid option!" << std::endl;
        return 1;
    }
  }

  if (verbose) {
    std::cout << "Path to dataset file: " << dataset_fname << std::endl;
    std::cout << "Path to query file: " << query_fname << std::endl;
    std::cout << "Path to groundtruth file: " << groundtruth_fname << std::endl;
    std::cout << "CAGRA graph degree: " << graph_degree << std::endl;
    std::cout << "Streaming build batch size: " << streaming_batch_size << std::endl;
    std::cout << "Top-K: " << topk << std::endl;
    std::cout << "Streaming build k-means training percentage: " << kmeans_percentage << std::endl;
    std::cout << "Enable data permutation for streaming build: " << (data_permutation ? "True" : "False") << std::endl;
    std::cout << "Streaming build train kmeans with beginning batches: " << (train_kmeans_with_beginning_batches ? "True" : "False") << std::endl;
    std::cout << "Skip running baseline CAGRA build: " << (skip_baseline ? "True" : "False") << std::endl;
    std::cout << "Allocation with host pinned memory: " << (allocate_pinned ? "True" : "False") << std::endl;
  }

  uint32_t n_samples, n_queries, n_dim, max_k; 
  read_header(dataset_fname, n_samples, n_dim);
  read_header(query_fname, n_queries, n_dim);
  read_header(groundtruth_fname, n_queries, max_k);
  size_t dataset_size = (size_t)n_samples * (size_t)n_dim * sizeof(float);
  size_t query_size = (size_t)n_queries * (size_t)n_dim * sizeof(float);
  size_t groundtruth_size = (size_t)n_queries * (size_t)max_k * sizeof(uint32_t);

  raft::device_resources dev_resources;
  // Set pool memory resource with 40GiB initial pool size. All allocations use the same pool.
  raft::resource::set_workspace_to_pool_resource(dev_resources, 40 * 1024 * 1024 * 1024ull);

  float *dataset = nullptr;
  float *query = nullptr;
  uint32_t *groundtruth = nullptr;

  if (allocate_pinned) {
    gpuErrchk(cudaMallocHost((void**)&dataset, dataset_size));
    gpuErrchk(cudaMallocHost((void**)&query, query_size));
    gpuErrchk(cudaMallocHost((void**)&groundtruth, groundtruth_size));
  } else {
    dataset = (float*)malloc(dataset_size);
    query = (float*)malloc(query_size);
    groundtruth = (uint32_t*)malloc(groundtruth_size);
  }

  bool success =  read_data(dataset_fname, dataset, n_samples, n_dim) && 
                  read_data(query_fname, query, n_queries, n_dim) &&
                  read_data(groundtruth_fname, groundtruth, n_queries, max_k);
  if (!success) {
    std::cout << "Data reading failed.\n";
    exit(1);
  }

  cagra::index_params index_params;
  index_params.graph_degree = graph_degree; 
  index_params.intermediate_graph_degree = graph_degree * 2;

  std::list<size_t> itopk_list = {16, 32, 64, 128, 256, 512};
  std::vector<float> recall_baseline; 
  std::vector<float> recall_streaming; 

  if (data_permutation) {
    // Permutation to make sure the dataset is not ordered. 
    float *dataset_permu = nullptr;
    uint32_t *groundtruth_permu = nullptr;
    if (allocate_pinned) {
      gpuErrchk(cudaMallocHost((void**)&dataset_permu, dataset_size));
      gpuErrchk(cudaMallocHost((void**)&groundtruth_permu, groundtruth_size));
    } else {
      dataset_permu = (float*)malloc(dataset_size);
      groundtruth_permu = (uint32_t*)malloc(groundtruth_size);
    }
    
    std::vector<size_t> permutation_idx(n_samples);
    // Fill in numbers from 0 to dataset.extent(0)-1 and then generate permutation indices by randomly shuflle the array. 
    std::iota(permutation_idx.begin(), permutation_idx.end(), 0); 
    std::mt19937 g(12345);
    std::shuffle(permutation_idx.begin(), permutation_idx.end(), g);
    // Copy data to the permutated location. 
    for (int64_t i = 0; i < n_samples; i++) {
      int64_t out_loc = permutation_idx[i];
      for (int64_t j = 0; j < n_dim; j++) { 
        dataset_permu[out_loc * n_dim + j] = dataset[i * n_dim + j];
      }
    }
    // Reassign groundtruth neighbors.
    for (int64_t i = 0; i < n_queries; i++) {
      for (int64_t j = 0; j < max_k; j++) {
        groundtruth_permu[i * max_k + j] = permutation_idx[groundtruth[i * max_k + j]];
      }
    }

    std::swap(dataset_permu, dataset);
    std::swap(groundtruth_permu, groundtruth);

    if (allocate_pinned) {
      gpuErrchk(cudaFreeHost(dataset_permu));
      gpuErrchk(cudaFreeHost(groundtruth_permu));
    } else {
      free(dataset_permu);
      free(groundtruth_permu);
    }
  }

  auto dataset_view = raft::make_host_matrix_view<float, int64_t>(dataset, n_samples, n_dim);
  auto query_view = raft::make_host_matrix_view<float, int64_t>(query, n_queries, n_dim);
  auto groundtruth_view = raft::make_host_matrix_view<uint32_t, int64_t>(groundtruth, n_queries, max_k);

  if (!skip_baseline) {
    baseline_build_search(dev_resources, itopk_list, recall_baseline, dataset_view, 
    query_view, 
    groundtruth_view, 
    index_params, topk, verbose);
  }

  streaming_build_search(
    dev_resources, 
    itopk_list, recall_streaming,
    dataset_view, 
    query_view, 
    groundtruth_view, 
    index_params, topk, streaming_batch_size, kmeans_percentage, train_kmeans_with_beginning_batches, verbose);
  
  if (!skip_baseline) {
    std::cout << "Recall difference (streaming CAGRA vs CAGRA),";
    for (int64_t i = 0; i < recall_baseline.size(); i++) {
      std::cout << recall_streaming[i] / recall_baseline[i] << ",";
    }
    std::cout << std::endl;
  }

  if (allocate_pinned) {
    gpuErrchk(cudaFreeHost(dataset));
    gpuErrchk(cudaFreeHost(groundtruth));
    gpuErrchk(cudaFreeHost(query));
  } else {
    free(dataset);
    free(groundtruth);
    free(query);
  }

  return 0; 
}
