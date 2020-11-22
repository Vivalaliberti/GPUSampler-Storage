#ifndef GRAPHLEARN_CORE_GRAPH_STORAGE_GPU_AUTO_INDEXING_H_
#define GRAPHLEARN_CORE_GRAPH_STORAGE_GPU_AUTO_INDEXING_H_

#include <cstdint>
#include "Types.h"
#include "slab_hash.cuh"

class GPUAutoIndexing {
public:
    GPUAutoIndexing(){
        IdType seed_ = time(nullptr);
        bool identity_hash_ = true;
        IndexType device_idx_ = 0;
        dynamic_allocator_ = new DynamicAllocatorT();
        num_buckets_ = 100000;
        slab_hash_ = new GpuSlabHash<IdType, IndexType, SlabHashTypeT::ConcurrentMap>(num_buckets_, dynamic_allocator_, device_idx_, seed_, identity_hash_);
        indexing_size_ = 0;
    }
    ~GPUAutoIndexing() = default;

    void Add(IdType id){

        IdType* single_id_list;
        cudaMallocManaged(&single_id_list, sizeof(IdType));
        single_id_list[0] = id;
        IndexType* single_index_list;
        cudaMallocManaged(&single_index_list, sizeof(IndexType));
        single_index_list[0] = indexing_size_;
        IndexType* single_exist_index_list;
        cudaMallocManaged(&single_exist_index_list, sizeof(IndexType));
        //single_exist_index_list[0] = -1;
        //std::cout<<"id "<<single_id_list[0]<<"\n";
        slab_hash_ -> searchIndividual(single_id_list, single_exist_index_list, 1);
        cudaDeviceSynchronize();
        //std::cout<<"exist "<<single_exist_index_list[0]<<"\n";
        if(single_exist_index_list[0] == -1){
            slab_hash_ -> buildBulk(single_id_list, single_index_list, 1);
            indexing_size_ ++;
        }
        cudaDeviceSynchronize();
        cudaFree(single_id_list);
        cudaFree(single_index_list);
        cudaFree(single_exist_index_list);
    }

    void buildBulk(IdType* h_key, IndexType* h_value, IndexType num_keys, IndexType num_buckets){
        IdType* d_key;
        IndexType* d_value;
        cudaMalloc(&d_key, num_keys * sizeof(IdType));
        cudaMalloc(&d_value, num_keys * sizeof(IndexType));
        cudaMemcpy(d_key, h_key, num_keys * sizeof(IdType), cudaMemcpyDefault);
        cudaMemcpy(d_value, h_value, num_keys * sizeof(IndexType), cudaMemcpyDefault);
        // IdType seed_ = time(nullptr);
        // bool identity_hash_ = true;
        // IndexType device_idx_ = 0;
        // dynamic_allocator_ = new DynamicAllocatorT();
        // slab_hash_ = new GpuSlabHash<IdType, IndexType, SlabHashTypeT::ConcurrentMap>(num_buckets, dynamic_allocator_, device_idx_, seed_, identity_hash_);
        slab_hash_->buildBulk(d_key, d_value, num_keys);
        cudaDeviceSynchronize();
        indexing_size_ += num_keys;
        cudaFree(d_key);
        cudaFree(d_value);
    }

    IndexType* Get(IdType* d_query, IndexType num_queries){
        IndexType* d_result;
        cudaMallocManaged(&d_result, num_queries * sizeof(IndexType));
        slab_hash_->searchIndividual(d_query, d_result, num_queries);//try searchBulk
        //std::cout<<d_result[0]<<"ss\n";
        cudaDeviceSynchronize();
        return d_result;
    }

private:
    //allocate GPU SlabHash
    DynamicAllocatorT* dynamic_allocator_;
    GpuSlabHash<IdType, IndexType, SlabHashTypeT::ConcurrentMap>* slab_hash_;
    IdType seed_;
	bool identity_hash_;
    IndexType device_idx_;
    IdType indexing_size_;
    IndexType num_buckets_;

};


#endif

