import subprocess
import math
import re

data_paths = ["../data/gist-960-euclidean/base.fbin", "../data/wiki_all_1M/base.1M.fbin"]
query_paths = ["../data/gist-960-euclidean/query.fbin", "../data/wiki_all_1M/queries.fbin"]
groundtruth_paths = ["../data/gist-960-euclidean/groundtruth.neighbors.ibin", "../data/wiki_all_1M/groundtruth.1M.neighbors.ibin"]
dims = [960, 768]
batch_data_sizes = [128000, 500000, 2000000, 8000000, 32000000, 128000000] # assuming floating point data type
recall_pattern = r"Recall difference \(streaming CAGRA vs CAGRA\),([\d\.]+),([\d\.]+),([\d\.]+),([\d\.]+),"
build_time_pattern = r"Build elapsed time: (\d+) milliseconds"

for i in range(len(data_paths)):
    data_path = data_paths[i]
    query_path = query_paths[i]
    groundtruth_path = groundtruth_paths[i]
    dim = dims[i]

    for j in range(len(batch_data_sizes)):
        batch_data_size = batch_data_sizes[j]
        batch_size = math.ceil(batch_data_size / dim)

        # Default 20% for training with the beginning batches; pinned host memory; graph degree 32; k = 10.
        item_str = "-p 0.2 -d " + str(data_path) + " -q " + str(query_path) + " -t " + str(groundtruth_path) + " -g 32 -k 10 " + "-b " + str(batch_size) + " -B -V"
        try:
            result = subprocess.run(f"./examples/cpp/build/STREAMING_CAGRA_EXAMPLE {item_str}", shell=True, capture_output=True, text=True, check=True)
            # print(result)
            recall_match = re.search(recall_pattern, result.stdout)
            build_time_matches = re.findall(build_time_pattern, result.stdout)
            if recall_match:
                recall_values = [float(recall_match.group(i)) for i in range(1, 5)]
            else:
                recall_values = []

            build_times = [int(time) for time in build_time_matches]
            print("Input options,", item_str)
            print("Recall diff values,", ', '.join(map(str, recall_values)))
            print("Build runtimes (ms),", ', '.join(map(str, build_times)))

        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running './examples/cpp/build/STREAMING_CAGRA_EXAMPLE' with argument '{item_str}': {e.stderr}")
