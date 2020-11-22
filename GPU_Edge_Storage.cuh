#ifndef GRAPHLEARN_CORE_GRAPH_STORAGE_GPU_EDGE_STORAGE_H_
#define GRAPHLEARN_CORE_GRAPH_STORAGE_GPU_EDGE_STORAGE_H_

#include <cstdint>
#include "Types.h"
#define LOAD_FACTOR 0.75
#define EXPAND_FACTOR 2
#include "Element_value.h"

class GPUEdgeStorage {
public:
    virtual ~GPUEdgeStorage() = default;

    virtual void SetSideInfo(const SideInfo* info) = 0;
    virtual const SideInfo* GetSideInfo() const = 0;

    /// Do some re-organization after data fixed.
    //virtual void Build() = 0;

    /// Get the total edge count after data fixed.
    virtual IdType Size() const = 0;
    virtual IdType Capacity() const = 0;
    /// An EDGE is made up of [ src_id, dst_id, weight, label, attributes ].
    /// Insert the value to get an unique id.
    /// If the value is invalid, return -1.
    virtual IdType Add(EdgeValue* value) = 0;

    /// Lookup edge infos by edge_id, including
    ///    source node id,
    ///    destination node id,
    ///    edge weight,
    ///    edge label,
    ///    edge attributes
    // virtual IdType GetSrcId(IdType edge_id) const = 0;
    // virtual IdType GetDstId(IdType edge_id) const = 0;
    // virtual float GetWeight(IdType edge_id) const = 0;
    // virtual int32_t GetLabel(IdType edge_id) const = 0;
    // virtual Attribute GetAttribute(IdType edge_id) const = 0;

    /// For the needs of traversal and sampling, the data distribution is
    /// helpful. The interface should make it convenient to get the global data.
    ///
    /// Get all the source node ids, the count of which is the same with Size().
    /// These ids are not distinct.
    virtual IdType* GetAllSrcIds() const = 0;
    /// Get all the destination node ids, the count of which is the same with
    /// Size(). These ids are not distinct.
    virtual IdType* GetAllDstIds() const = 0;
    /// Get all weights if existed, the count of which is the same with Size().  
    virtual float* GetAllWeights() const = 0;
    /// Get all labels if existed, the count of which is the same with Size().
    virtual int32_t* GetAllLabels() const = 0;
    /// Get all attributes if existed, the count of which is the same with Size().
    // virtual const std::vector<Attribute>* GetAttributes() const = 0;
};

GPUEdgeStorage* NewGPUMemoryEdgeStorage();
//EdgeStorage* NewCompressedMemoryEdgeStorage();


#endif  // GRAPHLEARN_CORE_GRAPH_STORAGE_EDGE_STORAGE_H_