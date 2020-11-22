#include "Storage_mode.h"
#include "GPU_AdjacentMatrix.cuh"
#include "GPU_Topo_Statics.cuh"
#include "GPU_Topo_Storage.cuh"
#include "Config.h"
#include <iostream>

class GPUMemoryTopoStorage : public GPUTopoStorage {
    GPUMemoryTopoStorage() : adj_matrix_(nullptr), statics_(nullptr){
        src_indexing_ = new GPUAutoIndexing();
        if(IsDataDistributionEnabled()){
          dst_indexing_ = new GPUAutoIndexing();
          statics_ = new GPUTopoStatics(src_indexing_, dst_indexing_);
        }
    }

    virtual ~GPUMemoryTopoStorage(){
        delete adj_matrix_;
        delete statics_;
    }

    void Add(IdType edge_id, EdgeValue* edge) override {
        src_indexing_ -> Add(edge -> src_id);
        adj_matrix_ -> Add(edge_id, edge -> src_id, edge -> dst_id);
        if(IsDataDistributionEnabled()){
            dst_indexing_ -> Add(edge -> dst_id);
            statics_ -> Add(edge -> src_id, edge -> dst_id);
        }
    }

    IdType* GetNeighbors(IndexType src_index) const override {
        return adj_matrix_ -> GetNeighbors(src_index);
    }

    IdType* GetOutEdges(IndexType src_index) const override {
        return adj_matrix_ -> GetOutEdges(src_index);
    }

    	//return the pointer of adjmatrix
	  IdType** GetNodeAdjMatrix() const override {
	  	  return adj_matrix_ -> GetNodeAdjMatrix();
    }
    
	  IdType** GetEdgeAdjMatrix() const override {
		    return adj_matrix_ -> GetEdgeAdjMatrix();
    }
    IdType GetRowSize() const override {
        return adj_matrix_ -> Row_Size();
    }
    IdType* GetColSize() const override {
        return adj_matrix_ -> Col_Size();
    }
    // IndexType GetInDegree(IndexType dst_index) const override {
    //     if(IsDataDistributionEnabled()){
    //         return statics_ -> GetInDegree(src_index);
    //     }else{
    //         return 0;
    //     }
    // }
    // IndexType GetOutDegree(IndexType src_index) const override {
    //     if(IsDataDistributionEnabled()){
    //         return statics_ -> GetOutDegree(src_index);
    //     }else{
    //         return 0;
    //     }
    // }

    IdType* GetAllSrcIds() const override {
        if(IsDataDistributionEnabled()){
            return statics_ -> GetAllSrcIds();
          }else{
            return nullptr;
          }
    }

    IdType* GetAllDstIds() const override {
        if(IsDataDistributionEnabled()){
            return statics_ -> GetAllDstIds();
          }else{
            return nullptr;
          }
    }

    IndexType* GetAllOutDegrees() const override {
        if(IsDataDistributionEnabled()){
            return statics_ -> GetAllOutDegrees();
          }else{
            return nullptr;
          }
    }

    IndexType* GetAllInDegrees() const override {
        if(IsDataDistributionEnabled()){
            return statics_ -> GetAllInDegrees();
          }else{
            return nullptr;
          }
    }

    IdType GetSrcIdSize() const override {
        if(IsDataDistributionEnabled()){
            return statics_ -> GetSrcIdSize();
        }else{
            return 0;
        }
    }

    IdType GetDstIdSize() const override {
        if(IsDataDistributionEnabled()){
            return statics_ -> GetDstIdSize();
        }else{
            return 0;
        }
    }

    IndexType* GetSrcAutoIndexing(IdType* query, IndexType num_queries) const override{
        return src_indexing_ -> Get(query, num_queries);
    }

    IndexType* GetDstAutoIndexing(IdType* query, IndexType num_queries) const override{
        if(IsDataDistributionEnabled()){
            return dst_indexing_ -> Get(query, num_queries);
        }else{
            return nullptr;
        }

    }

private:
    GPUAutoIndexing* src_indexing_;
    GPUAutoIndexing* dst_indexing_;
    GPUAdjMatrix* adj_matrix_;
    GPUTopoStatics* statics_;

    friend GPUTopoStorage* NewGPUMemoryTopoStorage();
    //compressed
};

GPUTopoStorage* NewGPUMemoryTopoStorage(){
    GPUMemoryTopoStorage* ret = new GPUMemoryTopoStorage();
    ret ->  adj_matrix_ = NewGPUMemoryAdjMatrix((ret -> src_indexing_));
    return ret;
}

//compressed