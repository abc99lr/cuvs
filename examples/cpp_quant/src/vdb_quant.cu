/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <string>
#include <algorithm>
#include <limits>

#include <raft/core/device_resources.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/stats/minmax.cuh>
#include <cuvs/neighbors/cagra.hpp>
#include <nvtx3/nvToolsExt.h>

#include <getopt.h>
#include "dataset.hpp"

#define CUDA_RT_CALL(call)                                                          \
{                                                                                   \
  cudaError_t cudaStatus = call;                                                    \
  if (cudaSuccess != cudaStatus) {                                                  \
    fprintf(stderr,                                                                 \
            "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "              \
            "with "                                                                 \
            "%s (%d).\n",                                                           \
            #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus); \
    exit( cudaStatus );                                                             \
  }                                                                                 \
}

enum class AlgoType {
  CAGRA,
  IVF_PQ,
  IVF_FLAT,
};

using namespace cuvs::neighbors;

// There is an implementation in RAFT for per-col quantization. The following kernels are useful for per-dataset quantization.
template<typename T>
__global__ void initialize_min_max(T* globalmin, T* globalmax) {
  *globalmin = std::numeric_limits<float>::max();
  *globalmax = std::numeric_limits<float>::min();
}

template <typename T>
__global__ void get_min_max(const T* dataset, int nrows, int ncols, T* globalmin, T* globalmax)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < nrows * ncols) {
    T val = dataset[tid];
    raft::myAtomicMin(globalmin, val);
    raft::myAtomicMax(globalmax, val);
  }
}

template<typename T>
__global__ void quantize_per_col(const T* dataset, uint8_t* quant_dataset, size_t nrows, size_t ncols, T* globalmin, T* globalmax)
{
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x; 
  if (tid < nrows * ncols) {
    size_t col_idx = tid / nrows;
    T start = globalmin[col_idx];
    T step = (globalmax[col_idx] - start) / 255;
    quant_dataset[tid] = static_cast<uint8_t>((dataset[tid] - start) / step); 
    // if (blockIdx.x == 0) {
    //   printf("tid %zu, start %lf, step %lf, dataset %lf, quant_dataset %c\n", tid, )
    // }
  }
}

template<typename NeighborType>
float cal_recall(const int32_t* gt_neighbors, NeighborType* res_neighbors, size_t number_queries, size_t topk, size_t max_topk) {
  float total_match = 0;
  float total_pred = number_queries * topk;
  for (size_t i = 0; i < number_queries; ++i) {
    for (size_t j = 0; j < topk; ++j) {
      int32_t gt_neighbor = gt_neighbors[i * max_topk + j];
      for (size_t k = 0; k < topk; ++k) {
        int32_t res_neighbor = static_cast<int32_t>(res_neighbors[i * topk + k]);
        if (gt_neighbor == res_neighbor) {
          total_match++; 
          break;
        }
      }
    }
    // std::vector<size_t> intersec_res(top_k);
    // std::sort(gt_neighbors + i * max_top_k, gt_neighbors + i * max_top_k + top_k);
    // std::sort(res_neighbors + i * top_k, res_neighbors + (i + 1) * top_k);
    // std::vector<size_t>::iterator it;
    // it = std::set_intersection(
    //   gt_neighbors + i * max_top_k, gt_neighbors + i * max_top_k + top_k,
    //   res_neighbors + i * top_k, res_neighbors + (i + 1) * top_k,
    //   intersec_res.begin()
    // );
    // total_correct += it - intersec_res.begin();
  }
  return total_match / total_pred;
}

template <typename T>
void run_main_cagra(raft::device_resources& dev_resources, std::shared_ptr<BinDataset<T>>& data, uint8_t* dataset_quant, bool quantization, MemoryType mem_location, cagra::index_params& index_params, cagra::search_params& search_params) {
  cudaEvent_t start, stop;
  CUDA_RT_CALL(cudaEventCreate(&start));
  CUDA_RT_CALL(cudaEventCreate(&stop));

  std::cout << "Building CAGRA index (search graph)" << std::endl;
  cagra::index<uint8_t, uint32_t> index_quant(dev_resources);
  // cagra::index<T, uint32_t> index_search(dev_resources);
  cagra::index<T, uint32_t> index(dev_resources);
  auto dataset_view = raft::make_device_matrix_view<const T, int64_t>(data->base_set(mem_location), data->base_set_size(), data->dim());

  nvtxRangePushA("cagra_build_baseline");
  CUDA_RT_CALL(cudaEventRecord(start));
  index = cagra::build(dev_resources, index_params, dataset_view);
  CUDA_RT_CALL(cudaEventRecord(stop));
  CUDA_RT_CALL(cudaEventSynchronize(stop));
  float build_time_ms = 0;
  CUDA_RT_CALL(cudaEventElapsedTime(&build_time_ms, start, stop));
  nvtxRangePop();

  std::cout << "CAGRA build time (s) " << build_time_ms / 1e3 << std::endl;
  std::cout << "CAGRA index has " << index.size() << " vectors" << std::endl;
  std::cout << "CAGRA graph has degree " << index.graph_degree() << ", graph size ["
            << index.graph().extent(0) << ", " << index.graph().extent(1) << "]" << std::endl;
  // index_search.update_dataset(dev_resources, index.dataset());
  // index_search.update_graph(dev_resources, index.graph());
  // std::cout << "metric " << index.metric() << " " << index_search.metric() << std::endl; 
  // std::cout << "size " << index.size() << " " << index_search.size() << std::endl; 
  // std::cout << "dim " << index.dim() << " " << index_search.dim() << std::endl; 
  // std::cout << "graph_degree " << index.graph_degree() << " " << index_search.graph_degree() << std::endl; 


  if (quantization) {
    auto dataset_view = raft::make_device_matrix_view<const T, int64_t>(data->base_set(mem_location), data->base_set_size(), data->dim());
    auto dataset_quant_view = raft::make_device_matrix_view<const uint8_t, int64_t>(dataset_quant, data->base_set_size(), data->dim());

    nvtxRangePushA("cagra_build_quant");
    CUDA_RT_CALL(cudaEventRecord(start));
    index_quant = cagra::build(dev_resources, index_params, dataset_quant_view);
    CUDA_RT_CALL(cudaEventRecord(stop));
    CUDA_RT_CALL(cudaEventSynchronize(stop));
    float build_time_ms = 0;
    CUDA_RT_CALL(cudaEventElapsedTime(&build_time_ms, start, stop));
    nvtxRangePop();

    std::cout << "CAGRA quant build time (s) " << build_time_ms / 1e3 << std::endl;
    std::cout << "CAGRA index has " << index_quant.size() << " vectors" << std::endl;
    std::cout << "CAGRA graph has degree " << index_quant.graph_degree() << ", graph size ["
              << index_quant.graph().extent(0) << ", " << index_quant.graph().extent(1) << "]" << std::endl;
    // index.update_dataset(dev_resources, index.dataset());
    index.update_graph(dev_resources, index_quant.graph());
    // std::cout << "metric " << index_quant.metric() << " " << index.metric() << std::endl; 
  } 

  size_t topk = 10;
  std::vector<size_t> itop_k_array{16, 32, 48, 64, 80, 96, 112, 128}; 
  auto query_view = raft::make_device_matrix_view<const T, int64_t>(data->query_set(mem_location), data->query_set_size(), data->dim());
  auto neighbors = raft::make_device_matrix<uint32_t>(dev_resources, data->query_set_size(), topk);
  auto distances = raft::make_device_matrix<float>(dev_resources, data->query_set_size(), topk);
  size_t total_num_search = 8;
  for (size_t i = 0; i < total_num_search; ++i) {
    search_params.itopk_size = itop_k_array[i];
    float search_time_ms = 0;
    CUDA_RT_CALL(cudaEventRecord(start));
    cagra::search(dev_resources, search_params, index, query_view, neighbors.view(), distances.view());
    CUDA_RT_CALL(cudaEventRecord(stop));
    CUDA_RT_CALL(cudaEventSynchronize(stop));
    CUDA_RT_CALL(cudaEventElapsedTime(&search_time_ms, start, stop));
    uint32_t *neighbors_host = nullptr;
    CUDA_RT_CALL(cudaMallocHost((void**)&neighbors_host, data->query_set_size() * topk * sizeof(uint32_t)));
    CUDA_RT_CALL(cudaMemcpy(neighbors_host, neighbors.data_handle(), data->query_set_size() * topk * sizeof(uint32_t), cudaMemcpyDefault));
    float recall = cal_recall(data->gt_set(), neighbors_host, data->query_set_size(), topk, data->max_k());
    float qps = data->query_set_size() / (search_time_ms / 1e3);
    std::cout << search_params.itopk_size << " " << qps << " "  << recall << " " << std::endl;
  }

  CUDA_RT_CALL(cudaEventDestroy(start));
  CUDA_RT_CALL(cudaEventDestroy(stop));

  return; 
}

template <typename T>
void run_main_ivf_pq(raft::device_resources& dev_resources, std::shared_ptr<BinDataset<T>>& data, uint8_t* dataset_quant, bool quantization, MemoryType mem_location, ivf_pq::index_params& index_params, ivf_pq::search_params& search_params) {
  cudaEvent_t start, stop;
  CUDA_RT_CALL(cudaEventCreate(&start));
  CUDA_RT_CALL(cudaEventCreate(&stop));

  std::cout << "Building IVF_PQ index" << std::endl;
  // cagra::index<uint8_t, uint32_t> index_quant(dev_resources);
  // cagra::index<T, uint32_t> index_search(dev_resources);
  // ivf_pq::index<T, int64_t> index(dev_resources);
  auto dataset_view = raft::make_device_matrix_view<const T, int64_t>(data->base_set(mem_location), data->base_set_size(), data->dim());
  auto dataset_quant_view = raft::make_device_matrix_view<const uint8_t, int64_t>(dataset_quant, data->base_set_size(), data->dim());

  nvtxRangePushA("ivf_pq_build_baseline");
  CUDA_RT_CALL(cudaEventRecord(start));
  auto index = ivf_pq::build(dev_resources, index_params, dataset_quant_view);
  CUDA_RT_CALL(cudaEventRecord(stop));
  CUDA_RT_CALL(cudaEventSynchronize(stop));
  float build_time_ms = 0;
  CUDA_RT_CALL(cudaEventElapsedTime(&build_time_ms, start, stop));
  nvtxRangePop();

  std::cout << "IVF-PQ build time (s) " << build_time_ms / 1e3 << std::endl;
  std::cout << "IVF-PQ index has " << index.size() << " vectors with dim " << index.dim() << std::endl;

 // if (quantization) {
  //   auto dataset_view = raft::make_device_matrix_view<const T, int64_t>(data->base_set(mem_location), data->base_set_size(), data->dim());
  //   auto dataset_quant_view = raft::make_device_matrix_view<const uint8_t, int64_t>(dataset_quant, data->base_set_size(), data->dim());

  //   nvtxRangePushA("ivf_pq_build_quant");
  //   CUDA_RT_CALL(cudaEventRecord(start));
  //   index_quant = ivf_pq::build(dev_resources, index_params, dataset_quant_view);
  //   CUDA_RT_CALL(cudaEventRecord(stop));
  //   CUDA_RT_CALL(cudaEventSynchronize(stop));
  //   float build_time_ms = 0;
  //   CUDA_RT_CALL(cudaEventElapsedTime(&build_time_ms, start, stop));
  //   nvtxRangePop();

  //   std::cout << "CAGRA quant build time (s) " << build_time_ms / 1e3 << std::endl;
  //   std::cout << "CAGRA index has " << index_quant.size() << " vectors" << std::endl;
  //   std::cout << "CAGRA graph has degree " << index_quant.graph_degree() << ", graph size ["
  //             << index_quant.graph().extent(0) << ", " << index_quant.graph().extent(1) << "]" << std::endl;
  //   // index.update_dataset(dev_resources, index.dataset());
  //   index.update_graph(dev_resources, index_quant.graph());
  //   // std::cout << "metric " << index_quant.metric() << " " << index.metric() << std::endl; 
  // } 

  // size_t topk = 10;
  // std::vector<size_t> n_probe_array{20, 30, 40, 50, 60, 80, 100, 200, 500}; 
  // auto query_view = raft::make_device_matrix_view<const T, int64_t>(data->query_set(mem_location), data->query_set_size(), data->dim());
  // // Why CAGRA and IVF-PQ uses different datatype for neighbors 
  // auto neighbors = raft::make_device_matrix<int64_t>(dev_resources, data->query_set_size(), topk);
  // auto distances = raft::make_device_matrix<float>(dev_resources, data->query_set_size(), topk);
  // size_t total_num_search = 9;
  // for (size_t i = 0; i < total_num_search; ++i) {
  //   nvtxRangePushA("ivf_pq_search");
  //   search_params.n_probes = n_probe_array[i];
  //   float search_time_ms = 0;
  //   CUDA_RT_CALL(cudaEventRecord(start));
  //   ivf_pq::search(dev_resources, search_params, index, query_view, neighbors.view(), distances.view());
  //   CUDA_RT_CALL(cudaEventRecord(stop));
  //   CUDA_RT_CALL(cudaEventSynchronize(stop));
  //   CUDA_RT_CALL(cudaEventElapsedTime(&search_time_ms, start, stop));
  //   int64_t *neighbors_host = nullptr;
  //   CUDA_RT_CALL(cudaMallocHost((void**)&neighbors_host, data->query_set_size() * topk * sizeof(int64_t)));
  //   CUDA_RT_CALL(cudaMemcpy(neighbors_host, neighbors.data_handle(), data->query_set_size() * topk * sizeof(int64_t), cudaMemcpyDefault));
  //   float recall = cal_recall(data->gt_set(), neighbors_host, data->query_set_size(), topk, data->max_k());
  //   float qps = data->query_set_size() / (search_time_ms / 1e3);
  //   std::cout << search_params.n_probes << " " << qps << " "  << recall << " " << std::endl;
  //   nvtxRangePop();
  // }

  CUDA_RT_CALL(cudaEventDestroy(start));
  CUDA_RT_CALL(cudaEventDestroy(stop));

  return; 
}

int main(int argc, char *argv[])
{
  // default arguments
  std::string dataset_path = ""; 
  std::string query_path = "";
  std::string gt_neighbors_path = "";
  bool quantization = false;
  MemoryType mem_location = MemoryType::Device; 
  bool print_header = false;
  AlgoType algo = AlgoType::CAGRA;

  // command line options
  const option long_opts[] = {
    {"dataset_path", required_argument, nullptr, 'd'},
    {"query_path", required_argument, nullptr, 'q'},
    {"gt_neighbors_path", required_argument, nullptr, 'g'},
    {"quantization", no_argument, nullptr, 'Q'},
    {"mem_location", required_argument, nullptr, 'l'},
    {"print_header", no_argument, nullptr, 'H'},
    {"algo", required_argument, nullptr, 'a'},
  };
  const std::string opts_desc[] = {
    "Path to input dataset in .{f/i}bin format. f/u8/i8 represents the input data type (float32, uint).",
    "Path to input query in .{f/i}bin format. f/u8/i8 represents the input data type (float32 or int).",
    "Path to input groundtruth eighbors in .ibin format.",
    "Enable quantization.",
    "Dataset location.",
    "Print header of reported performance.", 
    "VDB algorithm to test (0 - CAGRA, 1 - IVF_PQ, 2 - IVF_FLAT).",
  };
  const std::string opts_default[] = {
    dataset_path,
    query_path,
    gt_neighbors_path,
    std::to_string(quantization),
    std::to_string(static_cast<int>(mem_location)),
    std::to_string(print_header),
    std::to_string(static_cast<int>(algo)),
  };

  // parse command line
  int opt;
  while ((opt = getopt_long(argc, argv, "d:q:g:Ql:Ha:h", long_opts, nullptr)) != -1) {
    switch (opt) {
      case 'd': dataset_path = optarg; break;
      case 'q': query_path = optarg; break;
      case 'g': gt_neighbors_path = optarg; break;
      case 'Q': quantization = true; break;
      case 'l': mem_location = static_cast<MemoryType>(atoi(optarg)); break;
      case 'H': print_header = true; break; 
      case 'a': algo = static_cast<AlgoType>(atoi(optarg)); break;
      case 'h': {
        std::cout << "Usage:" << std::endl;
        int num_opts = std::extent<decltype(opts_desc)>::value;
        for (int i = 0; i < num_opts; i++)
        if (long_opts[i].has_arg != no_argument)
          std::cout << "  -" << (char)long_opts[i].val << ", --" << long_opts[i].name << " [arg]" << std::endl
          << "    " << opts_desc[i] << " [default: " << opts_default[i] << "]" << std::endl;
        else
          std::cout << "  -" << (char)long_opts[i].val << ", --" << long_opts[i].name << std::endl
          << "    " << opts_desc[i] << std::endl;
        exit(EXIT_FAILURE);
      }
      case '?':
        std::cout << "Please use -h or --help for the list of options" << std::endl;
        exit(EXIT_FAILURE);
      default:
        break;
    }
  }

  // Set pool memory resource with 1 GiB initial pool size. All allocations use the same pool.
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
    rmm::mr::get_current_device_resource(), 8 * 1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(&pool_mr);

  auto getExtension = [](const std::string& path) -> std::string {
    size_t pos = path.rfind('.');
    if (pos != std::string::npos) {
        return path.substr(pos);
    }
    return "";
  };

  std::string postfix = getExtension(dataset_path);
  std::cout << "Postfix " << postfix << std::endl;
  if (postfix == ".fbin") {
    // std::cout << "Insider runmain\n";
    auto data = std::make_shared<BinDataset<float>>(
      "vdb_quant",
      dataset_path,
      0,
      0,
      query_path,
      "euclidean",
      gt_neighbors_path
    );

    raft::device_resources dev_resources;
    std::cout << data->base_set_size() << " " << data->query_set_size() << " " << data->dim() << "\n";
    uint8_t *dataset_quant = nullptr;
    if (quantization) {
      CUDA_RT_CALL(cudaMalloc((void**)&dataset_quant, data->base_set_size() * data->dim() * sizeof(uint8_t)));
      float *min_device = nullptr, *max_device = nullptr; 
      float *min_host = nullptr, *max_host = nullptr; 
      CUDA_RT_CALL(cudaMalloc((void**)&min_device, data->dim() * sizeof(float)));
      CUDA_RT_CALL(cudaMalloc((void**)&max_device, data->dim() * sizeof(float)));
      CUDA_RT_CALL(cudaMallocHost((void**)&min_host, data->dim() * sizeof(float)));
      CUDA_RT_CALL(cudaMallocHost((void**)&max_host, data->dim() * sizeof(float)));

      // Per-dataset quantization 
      // initialize_min_max<<<1,1>>>(min_device, max_device);
      // get_min_max<<<ceil(1.0 * data->base_set_size() * data->dim() / 128), 128>>>(data->base_set(mem_location), data->base_set_size(), data->dim(), min_device, max_device);
      // CUDA_RT_CALL(cudaMemcpy(min_host, min_device, sizeof(float), cudaMemcpyDefault));
      // CUDA_RT_CALL(cudaMemcpy(max_host, max_device, sizeof(float), cudaMemcpyDefault));
      nvtxRangePushA("per_col_quant");
      // Per-col quantization 
      raft::stats::minmax<float>(data->base_set(mem_location), nullptr, nullptr, (int)data->base_set_size(), data->dim(), 1, min_device, max_device, nullptr, 0);
      quantize_per_col<<<ceil(1.0 * data->base_set_size() * data->dim() / 128), 128>>>(
        data->base_set(mem_location), dataset_quant, data->base_set_size(), data->dim(), min_device, max_device);
      nvtxRangePop();
      CUDA_RT_CALL(cudaFree(min_device));
      CUDA_RT_CALL(cudaFree(max_device));
      CUDA_RT_CALL(cudaFreeHost(min_host));
      CUDA_RT_CALL(cudaFreeHost(max_host));
    }

    if (algo == AlgoType::CAGRA) {
      cagra::index_params index_params;
      cagra::search_params search_params;
      run_main_cagra<float>(dev_resources, data, dataset_quant, quantization, mem_location, index_params, search_params);
    } else if (algo == AlgoType::IVF_PQ) {
      ivf_pq::index_params index_params;
      ivf_pq::search_params search_params;
      index_params.n_lists  = 1000;
      index_params.pq_dim = 128;
      index_params.pq_bits = 6;
      index_params.pq_bits = 6;
      index_params.kmeans_trainset_fraction = 0.2;
      run_main_ivf_pq<float>(dev_resources, data, dataset_quant, quantization, mem_location, index_params, search_params);
    } else {
      std::cout << "Unsupported algorithm\n";
      exit(1);
    }

    if (quantization) {
      CUDA_RT_CALL(cudaFree(dataset_quant));
    }
  } else {
    std::cout << "Unsupported data type\n";
    exit(1);
  }

  return 0;
}