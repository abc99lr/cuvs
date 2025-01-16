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

#include <string>
#include <fstream>
#include <iostream>

bool read_header(std::string const& data_fname, uint32_t& n_rows, uint32_t& n_cols) {
  std::ifstream file(data_fname, std::ifstream::binary);
  if (!file.is_open()) {
    std::cout << "Failed to open: " << data_fname << std::endl;
    return false;
  }
  file.read((char*)&n_rows, sizeof(uint32_t));
  file.read((char*)&n_cols, sizeof(uint32_t));
  if (file.fail()) {
    std::cerr << "Failed to read header." << std::endl;
    return false;
  }
  return true;
}

bool read_header(std::ifstream& file, uint32_t& n_rows, uint32_t& n_cols) {
  if (!file.is_open()) {
    std::cout << "Failed to read header." << std::endl;
    return false;
  }
  file.read((char*)&n_rows, sizeof(uint32_t));
  file.read((char*)&n_cols, sizeof(uint32_t));
  if (file.fail()) {
    std::cerr << "Failed to read header." << std::endl;
    return false;
  }
  return true;
}

template<typename DataT>
bool read_data(
  std::string const& data_fname, 
  DataT* h_data,
  uint32_t n_rows,
  uint32_t n_cols) {
  if (h_data == NULL) {
    std::cout << "Input data is not allocated" << std::endl;
    return false;
  }
  std::ifstream data_file(data_fname, std::ifstream::binary);
  if (!data_file.is_open()) {
    std::cout << "Failed to open: " << data_fname << std::endl;
    return false;
  }
  uint32_t n_rows_actual, n_cols_actual;
  if (!read_header(data_file, n_rows_actual, n_cols_actual)) {
    return false;
  }
  if (n_rows != n_rows_actual || n_cols != n_cols_actual) {
    std::cout << "Matrix size doesn't match with the actual size" << std::endl;
    return false;
  }
  if (!data_file.read((char*)h_data, n_rows * n_cols * sizeof(DataT))) {
    std::cout << "Failed to read: " << data_fname << std::endl;
    return false;
  }

  return true;
}
