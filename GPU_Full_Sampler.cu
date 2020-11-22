#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include "GPU_Graph_Storage.cuh"
#include "Types.h"
#include "Element_value.h"
#include "slab_hash.cuh"
#define batch_size_grained_thread_num 16
#define buffer_size_grained_thread_num 16

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

__global__ void kernel_get_neighbors_sizes(IndexType* d_src_indexs, IdType* source_col_size, IdType* return_col_size, IndexType batch_size){
	int ix = blockDim.x * blockIdx.x + threadIdx.x;
	int iy = blockDim.y * blockIdx.y + threadIdx.y;
	int idx = ix + (gridDim.x * blockDim.x) * iy;
	if (idx < batch_size){
		return_col_size[idx] = source_col_size[d_src_indexs[idx]];
	}
}

__global__ void kernel_set_positions(IdType* neighbor_position_buffer, IdType* cumulative_size_list, IndexType batch_size){
	int ix = blockDim.x * blockIdx.x + threadIdx.x;
	int iy = blockDim.y * blockIdx.y + threadIdx.y;
	int idx = ix + (gridDim.x * blockDim.x) * iy;
	if (idx < batch_size - 1){
		neighbor_position_buffer[cumulative_size_list[idx + 1]] = 1;
	}
}

__global__ void kernel_get_neighbors_ids(IndexType* d_src_indexs, IdType* neighbors_sizes_list, IdType** node_adjmatrix, IdType** edge_adjmatrix,
																  IdType* cumulative_size_list,
																  IdType* neighbor_position_buffer, IdType* GPU_dst_nodes_ids, IdType* GPU_dst_edges_ids, IdType buffer_size){
	int ix = blockDim.x * blockIdx.x + threadIdx.x;
	int iy = blockDim.y * blockIdx.y + threadIdx.y;
	int idx = ix + (gridDim.x * blockDim.x) * iy;
	if (idx < buffer_size){
		IdType position = neighbor_position_buffer[idx];
		GPU_dst_nodes_ids[idx] = node_adjmatrix[d_src_indexs[position]][idx - cumulative_size_list[position]];
		GPU_dst_edges_ids[idx] = edge_adjmatrix[d_src_indexs[position]][idx - cumulative_size_list[position]];
	}
}

class GPUFullSampler /* : public Sampler  */{
public:
	virtual ~GPUFullSampler() {}

	void StorageBuild(GPUGraphStorage* storage){
		storage_ = storage;
	}

  /*   Status Sample( const SamplingRequest* req,   SamplingResponse* res ) override { */
    void Sample(SamplingRequest* req) {			
		//settings
		cudaSetDevice(0);
		//settings
		IndexType batch_size = req->Batch_size();	
		IdType* h_src_ids = req->src_ids();

		//get src_indexs_list
		IdType* d_src_ids;
		cudaMallocManaged(&d_src_ids, batch_size * sizeof(IdType));
		cudaMemset(d_src_ids, 0, batch_size * sizeof(IdType));
		IndexType* d_src_indexs;
		cudaMallocManaged(&d_src_indexs, batch_size * sizeof(IndexType));
		cudaMemset(d_src_indexs, 0, batch_size * sizeof(IndexType));
		cudaMemcpy(d_src_ids, h_src_ids, batch_size * sizeof(IdType), cudaMemcpyHostToDevice);
		
		d_src_indexs = storage_ -> GetSrcAutoIndexing(d_src_ids, batch_size);
		cudaDeviceSynchronize();

		//get neighbors' sizes
		IdType* adjmatrix_col_size = storage_ -> GetAdjMatrixColSize();
		IdType* neighbors_sizes_list;
		cudaMallocManaged(&neighbors_sizes_list, batch_size * sizeof(IdType));
		cudaMemset(neighbors_sizes_list, 0, batch_size * sizeof(IdType));
		dim3 kernel_get_sizes_block(batch_size / (batch_size_grained_thread_num * batch_size_grained_thread_num) + 1, 1);
		dim3 kernel_get_sizes_thread(batch_size_grained_thread_num, batch_size_grained_thread_num);
		kernel_get_neighbors_sizes<<<kernel_get_sizes_block, kernel_get_sizes_thread>>>(d_src_indexs, adjmatrix_col_size, neighbors_sizes_list, batch_size);
		cudaDeviceSynchronize();
		for(int i = 0; i < batch_size; i++){
			std::cout<<"s "<<neighbors_sizes_list[i]<<"\n";
		}
		//get the sum of neighbors' sizes
		IdType buffer_size = thrust::reduce(neighbors_sizes_list, neighbors_sizes_list + batch_size);
		cudaDeviceSynchronize();
		std::cout<<"buffersize "<<buffer_size<<"\n";
		//exclusively scan the neighbor size list
		IdType* cumulative_size_list;
		cudaMallocManaged(&cumulative_size_list, batch_size * sizeof(IdType));
		cudaMemset(cumulative_size_list, 0, batch_size * sizeof(IdType));
		thrust::exclusive_scan(thrust::device, neighbors_sizes_list, neighbors_sizes_list + batch_size, cumulative_size_list, 0);
		cudaDeviceSynchronize();

		//set the position buffer
		IdType* neighbor_position_buffer;
		cudaMallocManaged(&neighbor_position_buffer, buffer_size * sizeof(IdType));
		cudaMemset(neighbor_position_buffer, 0, buffer_size * sizeof(IdType));
		cudaDeviceSynchronize();
		dim3 kernel_set_position_block(batch_size / (batch_size_grained_thread_num * batch_size_grained_thread_num) + 1, 1);
		dim3 kernel_set_position_thread(batch_size_grained_thread_num, batch_size_grained_thread_num);
		kernel_set_positions<<<kernel_set_position_block, kernel_set_position_thread>>>(neighbor_position_buffer, cumulative_size_list, batch_size);
		cudaDeviceSynchronize();

		thrust::inclusive_scan(thrust::device, neighbor_position_buffer, neighbor_position_buffer + buffer_size, neighbor_position_buffer);
		cudaDeviceSynchronize();

		//dst ids
		//IdType** node_adjmatrix = storage_ -> GetNodeAdjMatrix();
		IdType** node_adjmatrix = storage_ -> GetNodeAdjMatrix();
		IdType** edge_adjmatrix = storage_ -> GetEdgeAdjMatrix();
		IdType* GPU_dst_edges_ids;
		IdType* GPU_dst_nodes_ids;
		cudaMallocManaged(&GPU_dst_edges_ids, buffer_size * sizeof(IdType));
		cudaMallocManaged(&GPU_dst_nodes_ids, buffer_size * sizeof(IdType));
		cudaMemset(GPU_dst_edges_ids, 0, buffer_size * sizeof(IdType));
		cudaMemset(GPU_dst_nodes_ids, 0, buffer_size * sizeof(IdType));

		cudaDeviceSynchronize();
		dim3 kernel_get_neighbors_ids_block(buffer_size / (buffer_size_grained_thread_num * buffer_size_grained_thread_num) + 1, 1);
		dim3 kernel_get_neighbors_ids_thread(buffer_size_grained_thread_num, buffer_size_grained_thread_num);
		kernel_get_neighbors_ids<<<kernel_get_neighbors_ids_block, kernel_get_neighbors_ids_thread>>>(d_src_indexs, neighbors_sizes_list, node_adjmatrix, edge_adjmatrix, 
																										cumulative_size_list,
																										neighbor_position_buffer, GPU_dst_nodes_ids, GPU_dst_edges_ids, buffer_size);
		cudaDeviceSynchronize();

		for(int i = 0; i < buffer_size; i++){
			std::cout<<"n "<<GPU_dst_nodes_ids[i]<<" ";
			std::cout<<"e "<<GPU_dst_edges_ids[i]<<"\n";
		}
		//free storage
		cudaFree(GPU_dst_edges_ids);
		cudaFree(GPU_dst_nodes_ids);
		cudaFree(cumulative_size_list);
		cudaFree(neighbor_position_buffer);
		cudaFree(neighbors_sizes_list);
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