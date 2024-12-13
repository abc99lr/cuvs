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

package com.nvidia.cuvs.cagra;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import com.nvidia.cuvs.panama.CuVSCagraIndexParams;

/**
 * Supplemental parameters to build CAGRA Index.
 * 
 * @since 24.12
 */
public class CagraIndexParams {

  private final CagraGraphBuildAlgo cuvsCagraGraphBuildAlgo;
  private final MemorySegment memorySegment;
  private final Arena arena;
  private final int intermediateGraphDegree;
  private final int graphDegree;
  private final int nnDescentNiter;
  private final int numWriterThreads;

  /**
   * Enum that denotes which ANN algorithm is used to build CAGRA graph.
   */
  public enum CagraGraphBuildAlgo {
    /**
     * AUTO_SELECT
     */
    AUTO_SELECT(0),
    /**
     * IVF_PQ
     */
    IVF_PQ(1),
    /**
     * NN_DESCENT
     */
    NN_DESCENT(2);

    /**
     * The value for the enum choice.
     */
    public final int label;

    private CagraGraphBuildAlgo(int label) {
      this.label = label;
    }
  }

  private CagraIndexParams(Arena arena, int intermediateGraphDegree, int graphDegree,
      CagraGraphBuildAlgo CuvsCagraGraphBuildAlgo, int nnDescentNiter, int writerThreads) {
    this.arena = arena;
    this.intermediateGraphDegree = intermediateGraphDegree;
    this.graphDegree = graphDegree;
    this.cuvsCagraGraphBuildAlgo = CuvsCagraGraphBuildAlgo;
    this.nnDescentNiter = nnDescentNiter;
    this.numWriterThreads = writerThreads;

    this.memorySegment = initMemorySegment();
  }

  private MemorySegment initMemorySegment() {
    MemorySegment memorySegment = CuVSCagraIndexParams.allocate(arena);
    CuVSCagraIndexParams.intermediate_graph_degree(memorySegment, intermediateGraphDegree);
    CuVSCagraIndexParams.graph_degree(memorySegment, graphDegree);
    CuVSCagraIndexParams.build_algo(memorySegment, cuvsCagraGraphBuildAlgo.label);
    CuVSCagraIndexParams.nn_descent_niter(memorySegment, nnDescentNiter);
    return memorySegment;
  }

  /**
   * Gets the degree of input graph for pruning.
   * 
   * @return the degree of input graph
   */
  public int getIntermediateGraphDegree() {
    return intermediateGraphDegree;
  }

  /**
   * Gets the degree of output graph.
   * 
   * @return the degree of output graph
   */
  public int getGraphDegree() {
    return graphDegree;
  }

  /**
   * Gets the {@link CagraGraphBuildAlgo} used to build the index.
   */
  public CagraGraphBuildAlgo getCagraGraphBuildAlgo() {
    return cuvsCagraGraphBuildAlgo;
  }

  /**
   * Gets the number of iterations to run if building with {@link CagraGraphBuildAlgo#NN_DESCENT}
   */
  public int getNNDescentNumIterations() {
    return nnDescentNiter;
  }

  protected MemorySegment getMemorySegment() {
    return memorySegment;
  }

  @Override
  public String toString() {
    return "CagraIndexParams [arena=" + arena + ", intermediateGraphDegree=" + intermediateGraphDegree
        + ", graphDegree=" + graphDegree + ", cuvsCagraGraphBuildAlgo=" + cuvsCagraGraphBuildAlgo + ", nnDescentNiter="
        + nnDescentNiter + ", cagraIndexParamsMemorySegment=" + memorySegment + "]";
  }

  /**
   * Gets the number of threads used to build the index.
   */
  public int getNumWriterThreads() {
	return numWriterThreads;
}

  /**
   * Builder configures and creates an instance of {@link CagraIndexParams}.
   */
  public static class Builder {

    private CagraGraphBuildAlgo cuvsCagraGraphBuildAlgo = CagraGraphBuildAlgo.NN_DESCENT;
    private Arena arena;
    private int intermediateGraphDegree = 128;
    private int graphDegree = 64;
    private int nnDescentNumIterations = 20;
    private int numWriterThreads = 1;

    public Builder() {
    	
    }

    /**
     * Sets the degree of input graph for pruning.
     * 
     * @param intermediateGraphDegree degree of input graph for pruning
     * @return an instance of Builder
     */
    public Builder withIntermediateGraphDegree(int intermediateGraphDegree) {
      this.intermediateGraphDegree = intermediateGraphDegree;
      return this;
    }

    /**
     * Sets the degree of output graph.
     * 
     * @param graphDegree degree of output graph
     * @return an instance to Builder
     */
    public Builder withGraphDegree(int graphDegree) {
      this.graphDegree = graphDegree;
      return this;
    }

    /**
     * Sets the CuvsCagraGraphBuildAlgo to use.
     * 
     * @param cuvsCagraGraphBuildAlgo the CuvsCagraGraphBuildAlgo to use
     * @return an instance of Builder
     */
    public Builder withCagraGraphBuildAlgo(CagraGraphBuildAlgo cuvsCagraGraphBuildAlgo) {
      this.cuvsCagraGraphBuildAlgo = cuvsCagraGraphBuildAlgo;
      return this;
    }

    /**
     * Sets the Number of Iterations to run if building with {@link CagraGraphBuildAlgo#NN_DESCENT}.
     * 
     * @param nnDescentNiter number of Iterations to run if building with {@link CagraGraphBuildAlgo#NN_DESCENT}
     * @return an instance of Builder
     */
    public Builder withNNDescentNumIterations(int nnDescentNiter) {
      this.nnDescentNumIterations = nnDescentNiter;
      return this;
    }

    /**
     * Sets the number of writer threads to use for indexing.
     * 
     * @param numWriterThreads number of writer threads to use
     * @return an instance of Builder
     */
    public Builder withNumWriterThreads(int numWriterThreads) {
      this.numWriterThreads = numWriterThreads;
      return this;
    }

    /**
     * Builds an instance of {@link CagraIndexParams}.
     * 
     * @return an instance of {@link CagraIndexParams}
     */
    public CagraIndexParams build() {
      this.arena = Arena.ofConfined();
      return new CagraIndexParams(arena, intermediateGraphDegree, graphDegree, cuvsCagraGraphBuildAlgo, nnDescentNumIterations, numWriterThreads);
    }
  }
}