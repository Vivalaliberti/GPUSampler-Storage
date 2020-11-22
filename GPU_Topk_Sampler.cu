#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/linear_congruential_engine.h>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include "GPU_Graph_Storage.cuh"
#include "Types.h"
#include "Element_value.h"
#include "slab_hash.cuh"
#define batch_size_grained_thread_num 16
////////GLOBAL_FLAG(DefaultNeighborId) = 0/////////////
//using namespace std;

struct SamplingRequest{
	IndexType count = 5;
	IndexType batch_size = 32;
	IdType* Src_Ids= NULL;
	IndexType NeighborCount(){
		return count;
	}
	IndexType Batch_size(){
		return batch_size;
	}
	IdType* src_ids(){
		return Src_Ids;
	}
	
};

// __global__ void kernel_topk_sampler(IndexType* GPU_src_index, IdType** node_adjmatrix, IdType** edge_adjmatrix, IdType* all_col_size,
// 										IndexType batch_size, IndexType count, 
// 								IdType* GPU_dst_nodes_ids, IdType* GPU_dst_edges_ids){
// 	int ix = blockDim.x * blockIdx.x + threadIdx.x;
// 	int iy = blockDim.y * blockIdx.y + threadIdx.y;
// 	int idx = ix + (gridDim.x * blockDim.x) * iy;
// 	if (idx < batch_size){
// 		IdType GPU_src_index_ = GPU_src_index[idx];
// 		IdType* GPU_neighbor_ids = node_adjmatrix[GPU_src_index_];
// 		IdType* GPU_edge_ids = edge_adjmatrix[GPU_src_index_];
// 		IdType col_size = all_col_size[GPU_src_index_];
// 		IndexType min_count;
// 		if (count >= col_size){
// 			min_count = col_size;
// 		}else{
// 			min_count = count;
// 		}
// 		IndexType j;
// 		for (j = 0; j < min_count; j++) {
// 			GPU_dst_nodes_ids[count * idx + j] = GPU_neighbor_ids[j];
// 			GPU_dst_edges_ids[count * idx + j] = GPU_edge_ids[j];
// 		}
// 		for (j = min_count; j < count; j++){
// 			GPU_dst_nodes_ids[count * idx + j] = 0;
// 			GPU_dst_edges_ids[count * idx + j] = -1;
// 		}
// 	}
// }

__global__ void kernel_topk_sampler(IndexType* GPU_src_index, IdType** node_adjmatrix, IdType** edge_adjmatrix, IdType* all_col_size,
										IndexType batch_size, IndexType count, 
								IdType* GPU_dst_nodes_ids, IdType* GPU_dst_edges_ids){
	int ix = blockDim.x * blockIdx.x + threadIdx.x;
	int iy = blockDim.y * blockIdx.y + threadIdx.y;
	int idx = ix + (gridDim.x * blockDim.x) * iy;
	if (idx < batch_size){
		IdType GPU_src_index_ = GPU_src_index[idx];
		IdType* GPU_neighbor_ids = node_adjmatrix[GPU_src_index_];
		IdType* GPU_edge_ids = edge_adjmatrix[GPU_src_index_];
		IdType col_size = all_col_size[GPU_src_index_];

		IndexType j;
		IndexType i;
		for (i = 0, j = 0; j < count; j++, i++) {
			if (i == col_size){
				i = i - col_size;
			}
			GPU_dst_nodes_ids[count * idx + j] = GPU_neighbor_ids[i];
			GPU_dst_edges_ids[count * idx + j] = GPU_edge_ids[i];
		}
	}
}
class GPUTopkSampler /* : public Sampler  */{
public:
	virtual ~GPUTopkSampler() {}

	void StorageBuild(GPUGraphStorage* storage){
		storage_ = storage;
	}

  /*   Status Sample( const SamplingRequest* req,   SamplingResponse* res ) override { */
    void Sample(SamplingRequest* req) {			
		//settings
		cudaSetDevice(0);
		IndexType count = req->NeighborCount();
		IndexType batch_size = req->Batch_size();	
		IdType* h_src_ids = req->src_ids();
	
		IdType* d_src_ids;
		cudaMalloc(&d_src_ids, batch_size * sizeof(IdType));
		IndexType* d_src_indexs;
		cudaMalloc(&d_src_indexs, batch_size * sizeof(IndexType));
	
		cudaMemcpy(d_src_ids, h_src_ids, batch_size * sizeof(IdType), cudaMemcpyDefault);
		
		d_src_indexs = storage_ -> GetSrcAutoIndexing(d_src_ids, batch_size);
		cudaDeviceSynchronize();

		IdType** edge_adjmatrix = storage_ -> GetEdgeAdjMatrix();
		IdType** node_adjmatrix = storage_ -> GetNodeAdjMatrix();
		IdType* all_col_size = storage_ -> GetAdjMatrixColSize();
	
		//GPU kernel for sampling
		IndexType block_num;
		block_num = batch_size/(batch_size_grained_thread_num * batch_size_grained_thread_num)+1;
		dim3 kernel_sampling_block(block_num,1);
		dim3 kernel_sampling_thread(batch_size_grained_thread_num, batch_size_grained_thread_num);

		//dst ids
		IdType* GPU_dst_edges_ids;
		IdType* GPU_dst_nodes_ids;
		cudaMallocManaged(&GPU_dst_edges_ids, count * batch_size * sizeof(IdType));
		cudaMemset(GPU_dst_edges_ids, 0, count * batch_size * sizeof(IdType));
		cudaMallocManaged(&GPU_dst_nodes_ids, count * batch_size * sizeof(IdType));
		cudaMemset(GPU_dst_nodes_ids, 0, count * batch_size * sizeof(IdType));
		//sampling
		kernel_topk_sampler<<<kernel_sampling_block,kernel_sampling_thread>>>(d_src_indexs, node_adjmatrix, edge_adjmatrix, all_col_size, batch_size, count,  GPU_dst_nodes_ids, GPU_dst_edges_ids);
		cudaDeviceSynchronize();
		//IndexType* h_result = (IndexType*)malloc(batch_size * sizeof(IndexType));
		//cudaMemcpy(h_result, d_src_indexs, batch_size * sizeof(IndexType), cudaMemcpyDefault);
		for(int i = 0; i < count * batch_size; i++){
			std::cout<<"n "<<GPU_dst_nodes_ids[i]<<" ";
			std::cout<<"e "<<GPU_dst_edges_ids[i]<<"\n";
		}
		//free storage
		cudaFree(GPU_dst_edges_ids);
		cudaFree(GPU_dst_nodes_ids);
		cudaFree(d_src_indexs);
		//return Status::OK();
  }

private:
	GPUGraphStorage* storage_;
};

/* 	for (IndexType i = 0; i < batch_size; ++i) {
	  IdType src_id = src_ids[i];
      auto neighbor_ids = storage->GetNeighbors(src_id);
      if (!neighbor_ids) {
        res->FillWith(GLOBAL_FLAG(DefaultNeighborId), -1);
      } else {
        for (IndexType j = 0; j < count; ++j) {
          res->AppendNeighborId(dst_nodes_ids[count * (i - 1) + j]);
          res->AppendEdgeId(dst_edges_ids[count * (i - 1) + j]);
        }
    }
   }
	 */