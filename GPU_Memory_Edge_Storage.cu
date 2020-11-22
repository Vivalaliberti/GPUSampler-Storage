#include "GPU_Edge_Storage.cuh"
#include "Config.h"
#include <iostream>

class GPUMemoryEdgeStorage : public GPUEdgeStorage {
public:
    GPUMemoryEdgeStorage() {
        //int64_t estimate_size = GLOBAL_FLAG(AverageEdgeCount);
        int64_t estimate_size = 128;
        cudaMallocManaged(&src_ids_, estimate_size * sizeof(IdType));
        cudaMallocManaged(&dst_ids_, estimate_size * sizeof(IdType));
        //if(side_info_.IsWeighted()){
            cudaMallocManaged(&weights_, estimate_size * sizeof(IdType));
        //}
        //if(side_info_.IsLabeled()){
            cudaMallocManaged(&labels_, estimate_size * sizeof(IdType));
        //}
        //attribute
        edges_size_ = 0;
        edges_cap_ = estimate_size;
    }

    virtual ~GPUMemoryEdgeStorage() {
    }

    void SetSideInfo(const SideInfo* info) override {
        if (!side_info_.IsInitialized()) {
        side_info_.CopyFrom(*info);
        }
    }

    const SideInfo* GetSideInfo() const override {
        return &side_info_;
    }

    // void Build() override {
    //     src_ids_.shrink_to_fit();
    //     dst_ids_.shrink_to_fit();
    //     labels_.shrink_to_fit();
    //     weights_.shrink_to_fit();
    //     attributes_.shrink_to_fit();
    // }

    IdType Add(EdgeValue* value) override {
        IdType edge_id = edges_size_;
        if( edges_size_ + 1 >= LOAD_FACTOR * edges_cap_){
            Expand_All();
        }
        src_ids_[edge_id] = value -> src_id;
        dst_ids_[edge_id] = value -> dst_id;
        std::cout<<"src_id & dst_id "<<src_ids_[edge_id]<<" "<<dst_ids_[edge_id]<<"\n";
       // if (side_info_.IsWeighted()) {
            weights_[edge_id] = value -> weight;
            std::cout<<"weights["<<edge_id<<"]"<<weights_[edge_id]<<"\n";
       // }
       // if (side_info_.IsLabeled()) {
            labels_[edge_id] = value -> label;
            std::cout<<"labels["<<edge_id<<"]"<<labels_[edge_id]<<"\n";
        //}
        // if (side_info_.IsAttributed()) {
        // AttributeValue* attr = NewDataHeldAttributeValue();
        // attr->Swap(value->attrs);
        // attributes_.emplace_back(attr, true);
        // }
        edges_size_ ++;
        return edge_id;
    }

    IdType Size() const override {
        return edges_size_;
    }
    IdType Capacity() const override {
        return edges_cap_;
    }

    // IdType GetSrcId(IdType edge_id) const override {
    //     if (edge_id < Size()) {
    //     return src_ids_[edge_id];
    //     } else {
    //     return -1;
    //     }
    // }

    // IdType GetDstId(IdType edge_id) const override {
    //     if (edge_id < Size()) {
    //     return dst_ids_[edge_id];
    //     } else {
    //     return -1;
    //     }
    // }

    // float GetWeight(IdType edge_id) const override {
    //     if (edge_id < weights_.size()) {
    //     return weights_[edge_id];
    //     } else {
    //     return 0.0;
    //     }
    // }

    // int32_t GetLabel(IdType edge_id) const override {
    //     if (edge_id < labels_.size()) {
    //     return labels_[edge_id];
    //     } else {
    //     return -1;
    //     }
    // }

    // Attribute GetAttribute(IdType edge_id) const override {
    //     if (!side_info_.IsAttributed()) {
    //     return Attribute();
    //     }
    //     if (edge_id < attributes_.size()) {
    //     return Attribute(attributes_[edge_id].get(), false);
    //     } else {
    //     return Attribute(AttributeValue::Default(&side_info_), false);
    //     }
    // }

    IdType* GetAllSrcIds() const override {
        return src_ids_;
    }

    IdType* GetAllDstIds() const override {
        return dst_ids_;
    }

    float* GetAllWeights() const override {
        return weights_;
    }

    int32_t* GetAllLabels() const override {
        return labels_;
    }

    // const std::vector<Attribute>* GetAttributes() const override {
    //     return &attributes_;
    // }

private:
    IdType*     src_ids_;
    IdType      edges_size_;
    IdType      edges_cap_;
    IdType*     dst_ids_;
    int32_t*    labels_;
    float*      weights_;
    // std::vector<Attribute> attributes_;
    SideInfo    side_info_;

    void Expand_All(){
        std::cout<<"expand called\n";
        IdType* new_src_ids;
        IdType* new_dst_ids;
        int32_t* new_labels;
        float* new_weights;
        cudaMallocManaged(&new_src_ids, edges_cap_ * EXPAND_FACTOR * sizeof(IdType));
        cudaMallocManaged(&new_dst_ids, edges_cap_ * EXPAND_FACTOR * sizeof(IdType));
        cudaMallocManaged(&new_labels, edges_cap_ * EXPAND_FACTOR * sizeof(int32_t));
        cudaMallocManaged(&new_weights, edges_cap_ * EXPAND_FACTOR * sizeof(float));
        cudaMemset(new_src_ids, 0, edges_cap_ * EXPAND_FACTOR * sizeof(IdType));
        cudaMemset(new_dst_ids, 0, edges_cap_ * EXPAND_FACTOR * sizeof(IdType));
        cudaMemset(new_labels, 0, edges_cap_ * EXPAND_FACTOR * sizeof(int32_t));
        cudaMemset(new_weights, 0, edges_cap_ * EXPAND_FACTOR * sizeof(float));
        cudaMemcpy(new_src_ids, src_ids_, edges_size_ * sizeof(IdType), cudaMemcpyDefault);
        cudaMemcpy(new_dst_ids, dst_ids_, edges_size_ * sizeof(IdType), cudaMemcpyDefault);
        cudaMemcpy(new_labels, labels_, edges_size_ * sizeof(int32_t), cudaMemcpyDefault);
        cudaMemcpy(new_weights, weights_, edges_size_ * sizeof(float), cudaMemcpyDefault);
        cudaFree(src_ids_);
        cudaFree(dst_ids_);
        cudaFree(labels_);
        cudaFree(weights_);
        src_ids_ = new_src_ids;
        dst_ids_ = new_dst_ids;
        labels_ = new_labels;
        weights_ = new_weights;
        edges_cap_ *=EXPAND_FACTOR;
    }
};

GPUEdgeStorage* NewGPUMemoryEdgeStorage() {
    return new GPUMemoryEdgeStorage();
}