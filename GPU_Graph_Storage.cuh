#ifndef GRAPHLEARN_CORE_GRAPH_STORAGE_GPU_GRAPH_STORAGE_H_
#define GRAPHLEARN_CORE_GRAPH_STORAGE_GPU_GRAPH_STORAGE_H_

#include <cstdint>
#include <string>
#include "Types.h"
#include "Element_value.h"

class GPUGraphStorage {
public:
    virtual ~GPUGraphStorage() = default;

    // virtual void Lock() = 0;
    // virtual void Unlock() = 0;

    virtual void SetSideInfo(const SideInfo* info) = 0;
    virtual const SideInfo* GetSideInfo() const = 0;

    virtual void Add(EdgeValue* value) = 0;
    //virtual void Build() = 0;

    virtual IdType GetEdgeCount() const = 0;
    virtual IdType* GetAllSrcId() const = 0;
    virtual IdType* GetAllDstId() const = 0;
    virtual float* GetAllEdgeWeight() const = 0;
    virtual int32_t* GetAllEdgeLabel() const = 0;
    // virtual Attribute GetEdgeAttribute(IdType edge_id) const = 0;
    
    // virtual IdType GetSrcId(IdType edge_id) const = 0;
    // virtual IdType GetDstId(IdType edge_id) const = 0;
    // virtual float GetEdgeWeight(IdType edge_id) const = 0;
    // virtual int32_t GetEdgeLabel(IdType edge_id) const = 0;
    // virtual Attribute GetEdgeAttribute(IdType edge_id) const = 0;

    virtual IdType* GetNeighbors(IndexType src_index) const = 0;
    virtual IdType* GetOutEdges(IndexType src_index) const = 0;
    virtual IdType** GetNodeAdjMatrix() const = 0;
    virtual IdType** GetEdgeAdjMatrix() const = 0;
    virtual IdType GetAdjMatrixRowSize() const = 0;
    virtual IdType* GetAdjMatrixColSize() const = 0;
    //   virtual IndexType GetInDegree(IdType dst_id) const = 0;
    //   virtual IndexType GetOutDegree(IdType src_id) const = 0;
    virtual IndexType* GetAllInDegrees() const = 0;
    virtual IndexType* GetAllOutDegrees() const = 0;
    virtual IdType* GetAllSrcIds() const = 0;
    virtual IdType* GetAllDstIds() const = 0;
    virtual IdType GetSrcIdSize() const = 0;
    virtual IdType GetDstIdSize() const = 0;
    virtual IndexType* GetSrcAutoIndexing(IdType* query, IndexType num_queries) const = 0;
    virtual IndexType* GetDstAutoIndexing(IdType* query, IndexType num_queries) const = 0;
};

GPUGraphStorage* NewGPUMemoryGraphStorage();
//GPUGraphStorage* NewCompressedMemoryGraphStorage();


#endif  // GRAPHLEARN_CORE_GRAPH_STORAGE_GPU_GRAPH_STORAGE_H_