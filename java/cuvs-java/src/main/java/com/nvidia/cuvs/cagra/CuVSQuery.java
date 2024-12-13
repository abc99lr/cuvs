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

import java.util.Arrays;
import java.util.Map;

/**
 * CuVSQuery holds the CagraSearchParams and the query vectors to be used while
 * invoking search.
 * 
 * @since 24.12
 */
public class CuVSQuery {

  private CagraSearchParams cagraSearchParameters;
  private PreFilter preFilter;
  private Map<Integer, Integer> mapping;
  private float[][] queryVectors;
  private int topK;

  /**
   * Constructs an instance of CuVSQuery using cagraSearchParameters, preFilter,
   * queryVectors, mapping, and topK.
   * 
   * @param cagraSearchParameters an instance of CagraSearchParams holding the
   *                              search parameters
   * @param preFilter             an instance of PreFilter
   * @param queryVectors          2D float query vector array
   * @param mapping               an instance of ID mapping
   * @param topK                  the top k results to return
   */
  public CuVSQuery(CagraSearchParams cagraSearchParameters, PreFilter preFilter, float[][] queryVectors,
      Map<Integer, Integer> mapping, int topK) {
    super();
    this.cagraSearchParameters = cagraSearchParameters;
    this.preFilter = preFilter;
    this.queryVectors = queryVectors;
    this.mapping = mapping;
    this.topK = topK;
  }

  /**
   * Gets the instance of CagraSearchParams initially set.
   * 
   * @return an instance CagraSearchParams
   */
  public CagraSearchParams getCagraSearchParameters() {
    return cagraSearchParameters;
  }

  /**
   * Gets the instance of PreFilter initially set.
   * 
   * @return an instance of PreFilter
   */
  public PreFilter getPreFilter() {
    return preFilter;
  }

  /**
   * Gets the query vector 2D float array.
   * 
   * @return 2D float array
   */
  public float[][] getQueryVectors() {
    return queryVectors;
  }

  /**
   * Gets the passed map instance.
   * 
   * @return a map of ID mappings
   */
  public Map<Integer, Integer> getMapping() {
    return mapping;
  }

  /**
   * Gets the topK value.
   * 
   * @return an integer
   */
  public int getTopK() {
    return topK;
  }

  @Override
  public String toString() {
    return "CuVSQuery [cagraSearchParameters=" + cagraSearchParameters + ", preFilter=" + preFilter + ", queryVectors="
        + Arrays.toString(queryVectors) + ", mapping=" + mapping + ", topK=" + topK + "]";
  }

  /**
   * Builder helps configure and create an instance of CuVSQuery.
   */
  public static class Builder {

    private CagraSearchParams cagraSearchParams;
    private PreFilter preFilter;
    private float[][] queryVectors;
    private Map<Integer, Integer> mapping;
    private int topK = 2;

    /**
     * Default constructor.
     */
    public Builder() {
    }

    /**
     * Sets the instance of configured CagraSearchParams to be passed for search.
     * 
     * @param cagraSearchParams an instance of the configured CagraSearchParams to
     *                          be used for this query
     * @return an instance of this Builder
     */
    public Builder withSearchParams(CagraSearchParams cagraSearchParams) {
      this.cagraSearchParams = cagraSearchParams;
      return this;
    }

    /**
     * Registers the query vectors to be passed in the search call.
     * 
     * @param queryVectors 2D float query vector array
     * @return an instance of this Builder
     */
    public Builder withQueryVectors(float[][] queryVectors) {
      this.queryVectors = queryVectors;
      return this;
    }

    /**
     * Sets the PreFilter to be used with CuVSQuery.
     * 
     * @param preFilter the PreFilter instance to be configured
     * @return an instance of this Builder
     */
    public Builder withPreFilter(PreFilter preFilter) {
      this.preFilter = preFilter;
      return this;
    }

    /**
     * Sets the instance of mapping to be used for ID mapping.
     * 
     * @param mapping the ID mapping instance
     * @return an instance of this Builder
     */
    public Builder withMapping(Map<Integer, Integer> mapping) {
      this.mapping = mapping;
      return this;
    }

    /**
     * Registers the topK value.
     * 
     * @param topK the topK value used to retrieve the topK results
     * @return an instance of this Builder
     */
    public Builder withTopK(int topK) {
      this.topK = topK;
      return this;
    }

    /**
     * Builds an instance of CuVSQuery.
     * 
     * @return an instance of CuVSQuery
     */
    public CuVSQuery build() {
      return new CuVSQuery(cagraSearchParams, preFilter, queryVectors, mapping, topK);
    }
  }
}
