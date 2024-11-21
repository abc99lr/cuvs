package ivf_pq

import (
	"math/rand"
	"testing"
	"time"

	cuvs "github.com/rapidsai/cuvs/go"
)

func TestIvfPq(t *testing.T) {
	resource, _ := cuvs.NewResource(nil)

	rand.Seed(time.Now().UnixNano())

	NDataPoints := 1024
	NFeatures := 16

	TestDataset := make([][]float32, NDataPoints)
	for i := range TestDataset {
		TestDataset[i] = make([]float32, NFeatures)
		for j := range TestDataset[i] {
			TestDataset[i][j] = rand.Float32()
		}
	}

	dataset, _ := cuvs.NewTensor(TestDataset)

	IndexParams, err := CreateIndexParams()
	if err != nil {
		panic(err)
	}

	index, _ := CreateIndex(IndexParams, &dataset)
	defer index.Close()
	// use the first 4 points from the dataset as queries : will test that we get them back
	// as their own nearest neighbor

	NQueries := 4
	K := 4
	queries, _ := cuvs.NewTensor(TestDataset[:NQueries])
	NeighborsDataset := make([][]int64, NQueries)
	for i := range NeighborsDataset {
		NeighborsDataset[i] = make([]int64, K)
	}
	DistancesDataset := make([][]float32, NQueries)
	for i := range DistancesDataset {
		DistancesDataset[i] = make([]float32, K)
	}
	neighbors, _ := cuvs.NewTensor(NeighborsDataset)
	distances, _ := cuvs.NewTensor(DistancesDataset)

	_, todeviceerr := neighbors.ToDevice(&resource)
	if todeviceerr != nil {
		println(todeviceerr)
	}
	distances.ToDevice(&resource)
	dataset.ToDevice(&resource)

	err = BuildIndex(resource, IndexParams, &dataset, index)
	if err != nil {
		// println(err.Error())
		panic(err)
	}
	resource.Sync()

	queries.ToDevice(&resource)

	SearchParams, err := CreateSearchParams()
	if err != nil {
		panic(err)
	}

	err = SearchIndex(resource, SearchParams, index, &queries, &neighbors, &distances)
	if err != nil {
		panic(err)
	}

	neighbors.ToHost(&resource)
	distances.ToHost(&resource)

	resource.Sync()

	arr, _ := neighbors.Slice()
	for i := range arr {
		println(arr[i][0])
		if arr[i][0] != int64(i) {
			t.Error("wrong neighbor, expected", i, "got", arr[i][0])
		}
	}

	// arr_dist, _ := distances.GetArray()
	// for i := range arr_dist {
	// 	if arr_dist[i][0] >= float32(0.001) || arr_dist[i][0] <= float32(-0.001) {
	// 		t.Error("wrong distance, expected", float32(i), "got", arr_dist[i][0])
	// 	}
	// }
}
