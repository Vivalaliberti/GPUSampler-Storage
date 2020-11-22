//#include <mutex> 
#include <algorithm>
#include <functional>
#include <iostream>
#include "GPU_Graph_Storage.cuh"
#include "GPU_Edge_Storage.cuh"
#include "GPU_Topo_Storage.cuh"
#include "Config.h"
//#include "graphlearn/common/threading/sync/lock.h"

class GPUMemoryGraphStorage : public GPUGraphStorage {
public:
    GPUMemoryGraphStorage() {
        topo_ = NewGPUMemoryTopoStorage();
        edges_ = NewGPUMemoryEdgeStorage();
    }

    virtual ~GPUMemoryGraphStorage() {
        delete topo_;
        delete edges_;
    }

    // void Lock() override {
    //     mtx_.lock();
    // }

    // void Unlock() override {
    //     mtx_.unlock();
    // }

    void Add(EdgeValue* edge) override {
        IdType edge_id = edges_ -> Add(edge);
        std::cout<<"edgeid "<<edge_id<<"\n";
        if (edge_id != -1) {
          topo_ -> Add(edge_id, edge);
        }
    }
    
    // void Build() override {
    //     ScopedLocker<std::mutex> _(&mtx_);
    //     edges_->Build();
    //     topo_->Build(edges_);
    // }
    
    void SetSideInfo(const SideInfo* info) override {
        return edges_ -> SetSideInfo(info);
    }

    const SideInfo* GetSideInfo() const override {
        return edges_ -> GetSideInfo();
    }

    IdType GetEdgeCount() const override {
        return edges_ -> Size();
    }

    IdType* GetAllSrcId() const override {
        return edges_ -> GetAllSrcIds();
    }

    IdType* GetAllDstId() const override {
        return edges_ -> GetAllDstIds();
    }

    int32_t* GetAllEdgeLabel() const override {
        return edges_ -> GetAllLabels();
    }

    float* GetAllEdgeWeight() const override {
        return edges_ -> GetAllWeights();
    }


    // Attribute GetEdgeAttribute(IdType edge_id) const override {
    //     return edges_->GetAttribute(edge_id);
    // }

    IdType* GetNeighbors(IndexType src_index) const override {
        return topo_->GetNeighbors(src_index);
    }

    IdType* GetOutEdges(IndexType src_index) const override {
        return topo_->GetOutEdges(src_index);
    }

    IdType** GetNodeAdjMatrix() const override {
        return topo_ -> GetNodeAdjMatrix();
    }
    IdType** GetEdgeAdjMatrix() const override {
        return topo_ -> GetEdgeAdjMatrix();
    }

    IdType GetAdjMatrixRowSize() const override {
        return topo_ -> GetRowSize();
    }

    IdType* GetAdjMatrixColSize() const override {
        return topo_ -> GetColSize();
    }
    // IndexType GetInDegree(IdType dst_id) const override {
    //     return topo_->GetInDegree(dst_id);
    // }

    // IndexType GetOutDegree(IdType src_id) const override {
    //     return topo_->GetOutDegree(src_id);
    // }

    IndexType* GetAllInDegrees() const override {
        return topo_->GetAllInDegrees();
    }

    IndexType* GetAllOutDegrees() const override {
        return topo_->GetAllOutDegrees();
    }

    IdType* GetAllSrcIds() const override {
        return topo_->GetAllSrcIds();
    }

    IdType* GetAllDstIds() const override {
        return topo_->GetAllDstIds();
    }

    IdType GetSrcIdSize() const override {
        return topo_ -> GetSrcIdSize();
    }

    IdType GetDstIdSize() const override {
        return topo_ -> GetDstIdSize();
    }
    
    IndexType* GetSrcAutoIndexing(IdType* query, IndexType num_queries) const override{
        return topo_ -> GetSrcAutoIndexing(query, num_queries);
    }

    IndexType* GetDstAutoIndexing(IdType* query, IndexType num_queries) const override{
        return topo_ -> GetDstAutoIndexing(query, num_queries);
    }
    
private:
   // std::mutex   mtx_;
    GPUEdgeStorage* edges_;
    GPUTopoStorage* topo_;
};

GPUGraphStorage* NewGPUMemoryGraphStorage(){
    return new GPUMemoryGraphStorage();
}