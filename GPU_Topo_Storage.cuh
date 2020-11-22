#ifndef GRAPHLEARN_CORE_GRAPH_STORAGE_GPU_TOPO_STORAGE_H_
#define GRAPHLEARN_CORE_GRAPH_STORAGE_GPU_TOPO_STORAGE_H_

#include <cstdint>
#include <string>
#include "GPU_Edge_Storage.cuh"
#include "Types.h"

class GPUTopoStorage {
public:
    virtual ~GPUTopoStorage() = default;
    virtual void Add(IdType edge_id, EdgeValue* value) = 0;
    virtual IdType* GetNeighbors(IndexType src_index) const = 0;
    virtual IdType* GetOutEdges(IndexType src_index) const = 0;
    virtual IdType** GetNodeAdjMatrix() const = 0;
    virtual IdType** GetEdgeAdjMatrix() const = 0;
    virtual IdType GetRowSize() const = 0;
    virtual IdType* GetColSize() const = 0;
    // virtual IndexType GetInDegree(IndexType dst_index) const = 0;
    // virtual IndexType GetOutDegree(IndexType src_index) const = 0;
    virtual IdType* GetAllSrcIds() const = 0;//IdList*
    virtual IdType* GetAllDstIds() const = 0;
    virtual IndexType* GetAllOutDegrees() const = 0;//IndexList*
    virtual IndexType* GetAllInDegrees() const = 0;  
    virtual IdType GetSrcIdSize() const = 0;
    virtual IdType GetDstIdSize() const = 0;  
    virtual IndexType* GetSrcAutoIndexing(IdType* query, IndexType num_queries) const = 0;
    virtual IndexType* GetDstAutoIndexing(IdType* query, IndexType num_queries) const = 0;
};
GPUTopoStorage* NewGPUMemoryTopoStorage();
//compressed

#endif
