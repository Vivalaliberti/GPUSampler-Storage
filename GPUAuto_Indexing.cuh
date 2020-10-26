#ifndef GRAPHLEARN_CORE_GRAPH_STORAGE_GPU_AUTO_INDEXING_H_
#define GRAPHLEARN_CORE_GRAPH_STORAGE_GPU_AUTO_INDEXING_H_
#include "Types.h"
#include "slab_hash.cuh"

class GPUAutoIndexing {
public:
    GPUAutoIndexing(){}
    ~GPUAutoIndexing() = default;

    void buildBulk(IdType* h_key, IndexType* h_value, IndexType num_keys, IndexType num_buckets){
        IdType* d_key;
        IndexType* d_value;
        cudaMalloc(&d_key, num_keys * sizeof(IdType));
        cudaMalloc(&d_value, num_keys * sizeof(IndexType));
        cudaMemcpy(d_key, h_key, num_keys * sizeof(IdType), cudaMemcpyDefault);
        cudaMemcpy(d_value, h_value, num_keys * sizeof(IndexType), cudaMemcpyDefault);
        IdType seed_ = time(nullptr);
        bool identity_hash_ = true;
        IndexType device_idx_ = 0;
        dynamic_allocator_ = new DynamicAllocatorT();
        slab_hash_ = new GpuSlabHash<IdType, IndexType, SlabHashTypeT::ConcurrentMap>(num_buckets, dynamic_allocator_, device_idx_, seed_, identity_hash_);
        slab_hash_->buildBulk(d_key, d_value, num_keys);
        cudaFree(d_key);
    }

    IndexType* Get(IdType* d_query, IndexType num_queries){
        IndexType* d_result;
        cudaMalloc(&d_result, num_queries * sizeof(IndexType));
        slab_hash_->searchIndividual(d_query, d_result, num_queries);
        return d_result;
    }

private:
    //allocate GPU SlabHash
    DynamicAllocatorT* dynamic_allocator_;
    GpuSlabHash<IdType, IndexType, SlabHashTypeT::ConcurrentMap>* slab_hash_;
    IdType seed_;
	bool identity_hash_;
	IndexType device_idx_;
};


#endif