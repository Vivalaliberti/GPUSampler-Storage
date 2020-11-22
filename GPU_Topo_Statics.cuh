#ifndef GRAPHLEARN_CORE_GRAPH_STORAGE_GPU_TOPO_STATICS_H_
#define GRAPHLEARN_CORE_GRAPH_STORAGE_GPU_TOPO_STATICS_H_

#include <cstdint>
#include "Types.h"
#include "GPU_Auto_Indexing.cuh"
#define LOAD_FACTOR 0.75
#define EXPAND_FACTOR 2
#define topostatics_init_cap 128 


class GPUTopoStatics {
public:
    GPUTopoStatics(GPUAutoIndexing* src_indexing, GPUAutoIndexing* dst_indexing){
        src_indexing_ = src_indexing;
        dst_indexing_ = dst_indexing;
        src_id_size_ = 0;
        src_id_cap_ = topostatics_init_cap;
        cudaMallocManaged(&src_id_list_, src_id_cap_ * sizeof(IdType));
        cudaMallocManaged(&out_degree_list_, src_id_cap_ * sizeof(IndexType));
        cudaMemset(src_id_list_, 0, src_id_cap_ * sizeof(IdType));
        cudaMemset(out_degree_list_, 0, src_id_cap_ * sizeof(IndexType));
        dst_id_size_ = 0;
        dst_id_cap_ = topostatics_init_cap;
        cudaMallocManaged(&dst_id_list_, dst_id_cap_ * sizeof(IdType));
        cudaMallocManaged(&in_degree_list_, dst_id_cap_ * sizeof(IndexType));
        cudaMemset(dst_id_list_, 0, dst_id_cap_ * sizeof(IdType));
        cudaMemset(in_degree_list_, 0, dst_id_cap_ * sizeof(IdType));
    };

    virtual ~GPUTopoStatics() = default;

    //void Build();

    void Add(IdType src_id, IdType dst_id);

    IdType* GetAllSrcIds() const {
        return src_id_list_;
    }

    IdType GetSrcIdSize() const {
        return src_id_size_;
    }

    IdType* GetAllDstIds() const {
        return dst_id_list_;
    }

    IdType GetDstIdSize() const {
        return dst_id_size_;
    }

    IndexType* GetAllOutDegrees() const {
        return out_degree_list_;
    }

    IndexType* GetAllInDegrees() const {
        return in_degree_list_;
    }

    //IndexType GetOutDegree(IdType src_id) const;
    //IndexType GetInDegree(IdType dst_id) const;

private:
    GPUAutoIndexing* src_indexing_;
    GPUAutoIndexing* dst_indexing_;
    IdType*     src_id_list_;//IdList
    IdType      src_id_size_;
    IdType      src_id_cap_;
    IdType*     dst_id_list_;
    IdType      dst_id_size_;
    IdType      dst_id_cap_;
    IndexType*  out_degree_list_;//IndexList
    IndexType*  in_degree_list_;

    void Expand_Src(){
        std::cout<<"expand_src called\n";
        if (src_id_size_ + 1 >= LOAD_FACTOR * src_id_cap_){
            IdType* new_src_id_list;
            IndexType* new_out_degree_list;
            cudaMallocManaged(&new_src_id_list, EXPAND_FACTOR * src_id_cap_ * sizeof(IdType));
            cudaMallocManaged(&new_out_degree_list, EXPAND_FACTOR * src_id_cap_ * sizeof(IndexType));
            cudaMemset(new_src_id_list, 0, EXPAND_FACTOR * src_id_cap_ * sizeof(IdType));
            cudaMemset(new_out_degree_list, 0, EXPAND_FACTOR * src_id_cap_ * sizeof(IndexType));
            cudaMemcpy(new_src_id_list, src_id_list_, src_id_size_ * sizeof(IdType), cudaMemcpyDefault);
            cudaMemcpy(new_out_degree_list, out_degree_list_, src_id_size_ * sizeof(IndexType), cudaMemcpyDefault);
            cudaFree(src_id_list_);
            cudaFree(out_degree_list_);
            src_id_list_ = new_src_id_list;
            out_degree_list_ = new_out_degree_list;
            src_id_cap_ *= EXPAND_FACTOR;
        }
    }
    void Expand_Dst(){
        std::cout<<"expand_dst called\n";
        if (dst_id_size_ + 1 >= LOAD_FACTOR * dst_id_cap_){
            IdType* new_dst_id_list;
            IndexType* new_in_degree_list;
            cudaMallocManaged(&new_dst_id_list, EXPAND_FACTOR * dst_id_cap_ * sizeof(IdType));
            cudaMallocManaged(&new_in_degree_list, EXPAND_FACTOR * dst_id_cap_ * sizeof(IndexType));
            cudaMemset(new_dst_id_list, 0, EXPAND_FACTOR * dst_id_cap_ * sizeof(IdType));
            cudaMemset(new_in_degree_list, 0, EXPAND_FACTOR * dst_id_cap_ * sizeof(IndexType));
            cudaMemcpy(new_dst_id_list, dst_id_list_, dst_id_size_ * sizeof(IdType), cudaMemcpyDefault);
            cudaMemcpy(new_in_degree_list, in_degree_list_, dst_id_size_ * sizeof(IndexType), cudaMemcpyDefault);
            cudaFree(dst_id_list_);
            cudaFree(in_degree_list_);
            dst_id_list_ = new_dst_id_list;
            in_degree_list_ = new_in_degree_list;
            dst_id_cap_ *= EXPAND_FACTOR;
        }
    }
};


/*void TopoStatics::Build() {
    src_id_list_.shrink_to_fit();
    dst_id_list_.shrink_to_fit();
    out_degree_list_.shrink_to_fit();
    in_degree_list_.shrink_to_fit();
}*/

void GPUTopoStatics::Add(IdType src_id, IdType dst_id) {
    IdType* single_src_list;
    cudaMallocManaged(&single_src_list, sizeof(IdType));
    single_src_list[0] = src_id;
    IndexType* single_index_list;
    cudaMallocManaged(&single_index_list, sizeof(IdType));
    single_index_list = src_indexing_ -> Get(single_src_list, 1);
    cudaDeviceSynchronize();
    IndexType src_index = single_index_list[0];
    cudaFree(single_src_list);
    if (src_index < src_id_size_) {
        // has appeared before
        out_degree_list_[src_index]++;
        std::cout<<"out_degree_list["<<src_index<<"]"<<out_degree_list_[src_index]<<"\n";
    } else if (src_index == src_id_size_) {
        // new coming
        if (src_id_size_ + 1 >= LOAD_FACTOR * src_id_cap_){
            Expand_Src();
        }//load_factor = 0.75
        src_id_list_[src_index] = src_id;
        src_id_size_ ++;
        out_degree_list_[src_index] = 1;
        std::cout<<"out_degree_list["<<src_index<<"]"<<out_degree_list_[src_index]<<"\n";
    } else {
        // just ignore other cases
    }

    IdType* single_dst_list;
    cudaMallocManaged(&single_dst_list, sizeof(IdType));
    single_dst_list[0] = dst_id;
    single_index_list = dst_indexing_ -> Get(single_dst_list, 1);
    IndexType dst_index = single_index_list[0];
    cudaDeviceSynchronize();
    cudaFree(single_dst_list);
    if (dst_index < dst_id_size_) {
        // has appeared before
        in_degree_list_[dst_index]++;
        std::cout<<"in_degree_list["<<dst_index<<"]"<<in_degree_list_[dst_index]<<"\n";
    } else if (dst_index == dst_id_size_) {
        // new coming
        if (dst_id_size_ + 1 >= LOAD_FACTOR * dst_id_cap_){
            Expand_Dst();
        }//load_factor = 0.75
        dst_id_list_[dst_index] = dst_id;
        dst_id_size_ ++;
        in_degree_list_[dst_index] = 1;
        std::cout<<"in_degree_list["<<dst_index<<"]"<<in_degree_list_[dst_index]<<"\n";
    } else {
        // just ignore other cases
    }
}

/*
IndexType TopoStatics::GetOutDegree(IdType src_id) const {
    IndexType src_index = src_indexing_->Get(src_id);
    if (src_index < out_degree_list_.size()) {
        return out_degree_list_[src_index];
    } else {
        return 0;
    }
}

IndexType TopoStatics::GetInDegree(IdType dst_id) const {
    IndexType dst_index = dst_indexing_->Get(dst_id);
    if (dst_index < in_degree_list_.size()) {
        return in_degree_list_[dst_index];
    } else {
        return 0;
    }
}
*/

#endif