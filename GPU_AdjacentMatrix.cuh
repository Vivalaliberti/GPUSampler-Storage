#ifndef GRAPHLEARN_CORE_GRAPH_STORAGE_GPU_ADJ_MATRIX_H_
#define GRAPHLEARN_CORE_GRAPH_STORAGE_GPU_ADJ_MATRIX_H_

#include <cstdint>
#include "Types.h"
#include "GPU_Auto_Indexing.cuh"
#include "GPU_Edge_Storage.cuh"

//using namespace std;

#define init_row_cap 128
#define init_col_cap 16
#define LOAD_FACTOR 0.75
#define EXPAND_FACTOR 2
#define kernel_add_thread_num 8
#define kernel_expand_thread_num 8
#define kernel_repoint_thread_num 16
#define kernel_init_thread_num 16

__global__ void RePoint(IdType** new_adjmatrix, IdType** adj_nodes_, IdType GPU_Matrix_rows_);
__global__ void Init_Device_Array(IdType* array, IdType init_value, IndexType batch_size);

class GPUAdjMatrix{
public:
    virtual ~GPUAdjMatrix() = default;
    virtual IdType Row_Size() const = 0;
    virtual IdType Row_Cap() const = 0;
    virtual IdType* Col_Size() const = 0;
    virtual IdType* Col_Cap() const = 0;
    virtual void Add(IdType edge_id, IdType src_id, IdType dst_id) = 0;
    virtual IdType** GetNodeAdjMatrix() const = 0;
    virtual IdType** GetEdgeAdjMatrix() const = 0;
    virtual IdType* GetNeighbors(IndexType src_index) const = 0;
    virtual IdType* GetOutEdges(IndexType src_index) const = 0;

};

GPUAdjMatrix* NewGPUMemoryAdjMatrix(GPUAutoIndexing* indexing);

#endif   // GRAPHLEARN_CORE_GRAPH_STORAGE_GPU_ADJ_MATRIX_H_