#ifndef GRAPHLEARN_CORE_GRAPH_STORAGE_GPU_ADJ_MATRIX_H_
#define GRAPHLEARN_CORE_GRAPH_STORAGE_GPU_ADJ_MATRIX_H_
#include "Types.h"
#include <cuda.h>

using namespace std;

#define init_row_cap 128
#define init_col_cap 16
#define LOAD_FACTOR 0.75
#define expand_factor 2
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
    virtual void Add(IndexType src_index, IdType dst_id, IdType edge_id) = 0;
    virtual IdType** GetNodeAdjMatrix() const = 0;
    virtual IdType** GetEdgeAdjMatrix() const = 0;
    virtual IdType* GetNeighbors(IndexType src_index) const = 0;
    virtual IdType* GetOutEdges(IndexType src_index) const = 0;

};

#endif   // GRAPHLEARN_CORE_GRAPH_STORAGE_GPU_ADJ_MATRIX_H_