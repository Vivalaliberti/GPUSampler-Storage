#include <thrust/random/uniform_real_distribution.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include <vector>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include "Types.h"
#include "Element_value.h"
#include "GPU_Graph_Storage.cuh"
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

__global__ void kernel_get_neighbors_ids(IndexType* d_src_indexs, IdType* neighbors_sizes_list, IdType** edge_adjmatrix, 
																  IdType* cumulative_size_list,
																  IdType* neighbor_position_buffer, IdType* neighbor_id_buffer, IdType buffer_size){
	int ix = blockDim.x * blockIdx.x + threadIdx.x;
	int iy = blockDim.y * blockIdx.y + threadIdx.y;
	int idx = ix + (gridDim.x * blockDim.x) * iy;
	if (idx < buffer_size){
		IdType position = neighbor_position_buffer[idx];
		neighbor_id_buffer[idx] = edge_adjmatrix[d_src_indexs[position]][idx - cumulative_size_list[position]];
	}
}

__global__ void kernel_get_neighbors_weights(IdType* neighbor_id_buffer, float* neighbor_weight_buffer, float* all_edge_weights, IdType buffer_size){
	int ix = blockDim.x * blockIdx.x + threadIdx.x;
	int iy = blockDim.y * blockIdx.y + threadIdx.y;
	int idx = ix + (gridDim.x * blockDim.x) * iy;
	if (idx < buffer_size){
		neighbor_weight_buffer[idx] = all_edge_weights[neighbor_id_buffer[idx]];
	}
}
/*
__global__ void kernel_get_partial_sums(float* neighbor_weight_buffer, float* partial_sums, IdType* cumulative_size_list, IndexType batch_size, IdType buffer_size){
	int ix = blockDim.x * blockIdx.x + threadIdx.x;
	int iy = blockDim.y * blockIdx.y + threadIdx.y;
	int idx = ix + (gridDim.x * blockDim.x) * iy;
	if (idx < batch_size - 1){
		partial_sums[idx] = thrust::reduce(neighbor_weight_buffer + cumulative_size_list[idx], neighbor_weight_buffer + cumulative_size_list[idx + 1]);
		thrust::exclusive_scan(thrust::device, neighbor_weight_buffer + cumulative_size_list[idx], neighbor_weight_buffer + cumulative_size_list[idx + 1], neighbor_weight_buffer + cumulative_size_list[idx]);
	}else if(idx == batch_size - 1){
		partial_sums[idx] = thrust::reduce(neighbor_weight_buffer + cumulative_size_list[idx], neighbor_weight_buffer + buffer_size);
		thrust::exclusive_scan(thrust::device, neighbor_weight_buffer + cumulative_size_list[idx], neighbor_weight_buffer + buffer_size, neighbor_weight_buffer + cumulative_size_list[idx]);
	}
}*/

__global__ void kernel_get_cdf(float* neighbor_weight_buffer, float* partial_sums, IdType* neighbor_position_buffer, IdType buffer_size){
	int ix = blockDim.x * blockIdx.x + threadIdx.x;
	int iy = blockDim.y * blockIdx.y + threadIdx.y;
	int idx = ix + (gridDim.x * blockDim.x) * iy;
	if (idx < buffer_size){
		neighbor_weight_buffer[idx] = neighbor_weight_buffer[idx] / partial_sums[neighbor_position_buffer[idx]];
	}
}

__global__ void kernel_generate_random_number(float* random_number_buffer, IdType length){
	int ix = blockDim.x * blockIdx.x + threadIdx.x;
	int iy = blockDim.y * blockIdx.y + threadIdx.y;
	int idx = ix + (gridDim.x * blockDim.x) * iy;
	if (idx < length){
		thrust::minstd_rand engine;
		engine.discard(idx);
		thrust::uniform_real_distribution<float> dist(0, 1);
		random_number_buffer[idx] = dist(engine);
	}
}
//kernel return all the dst ids
__global__ void kernel_get_all_dst_ids(IdType* neighbor_position_buffer, float* neighbor_weight_buffer, IdType* neighbor_id_buffer,
																										float* random_number_buffer,
																				//IdType* GPU_dst_nodes_ids,
																				IdType* GPU_dst_edges_ids,
																				IndexType count, IndexType batch_size, IdType buffer_size){
	int ix = blockDim.x * blockIdx.x + threadIdx.x;
	int iy = blockDim.y * blockIdx.y + threadIdx.y;
	int idx = ix + (gridDim.x * blockDim.x) * iy;

	if (idx < buffer_size - 1){
		for(int i = 0; i < count; i++){
			IdType position = neighbor_position_buffer[idx];
			float random_number = random_number_buffer[position * count + i];
			if (neighbor_weight_buffer[idx] <= random_number && (neighbor_weight_buffer[idx + 1] > random_number || neighbor_weight_buffer[idx + 1] == 0)){
				GPU_dst_edges_ids[position * count + i] = neighbor_id_buffer[idx];
			}
		}
	} else if (idx == buffer_size - 1){
		for(int i = 0; i < count; i++){
			IdType position = neighbor_position_buffer[idx];
			float random_number = random_number_buffer[position * count + i];
			if (neighbor_weight_buffer[idx] <= random_number){
				GPU_dst_edges_ids[position * count + i] = neighbor_id_buffer[idx];
			}
		}
	}
}

class GPUEdgeWeightSampler /* : public Sampler  */{
public:
	virtual ~GPUEdgeWeightSampler() {}

	void StorageBuild(GPUGraphStorage* storage){
		storage_ = storage;
	}

  /*   Status Sample( const SamplingRequest* req,   SamplingResponse* res ) override { */
    void Sample(SamplingRequest* req) {		

		cudaSetDevice(0);
		//settings
		IndexType count = req->NeighborCount();
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
		for(int i = 0; i < batch_size; i++){
			std::cout<<"index "<<d_src_indexs[i]<<"\n";
		}
		//get neighbors' sizes
		IdType* adjmatrix_col_size = storage_ -> GetAdjMatrixColSize();
		IdType* neighbors_sizes_list;
		cudaMallocManaged(&neighbors_sizes_list, batch_size * sizeof(IdType));
		cudaMemset(neighbors_sizes_list, 0, batch_size * sizeof(IdType));
		dim3 kernel_get_sizes_block(batch_size / (batch_size_grained_thread_num * batch_size_grained_thread_num) + 1, 1);
		dim3 kernel_get_sizes_thread(batch_size_grained_thread_num, batch_size_grained_thread_num);
		kernel_get_neighbors_sizes<<<kernel_get_sizes_block, kernel_get_sizes_thread>>>(d_src_indexs, adjmatrix_col_size, neighbors_sizes_list, batch_size);
		cudaDeviceSynchronize();
		//get the sum of neighbors' sizes
		IdType buffer_size = thrust::reduce(neighbors_sizes_list, neighbors_sizes_list + batch_size);
		cudaDeviceSynchronize();
		std::cout<<"buffersize"<<buffer_size<<"\n";
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

		//allocate & fill the id/weight buffers
		//IdType** node_adjmatrix = storage_ -> GetNodeAdjMatrix();
		IdType** edge_adjmatrix = storage_ -> GetEdgeAdjMatrix();
		IdType* neighbor_id_buffer;
		cudaMallocManaged(&neighbor_id_buffer, buffer_size * sizeof(IdType));
		cudaMemset(neighbor_id_buffer, 0, buffer_size * sizeof(IdType));
		//IndexType* neighbor_index_buffer;
		//cudaMallocManaged(&neighbor_index_buffer, buffer_size * sizeof(IndexType));
		//cudaMemset(neighbor_index_buffer, 0, buffer_size * sizeof(IndexType));
		float* neighbor_weight_buffer;
		cudaMallocManaged(&neighbor_weight_buffer, buffer_size * sizeof(float));
		cudaMemset(neighbor_weight_buffer, 0, buffer_size * sizeof(float));
		cudaDeviceSynchronize();
		dim3 kernel_get_neighbors_ids_block(buffer_size / (buffer_size_grained_thread_num * buffer_size_grained_thread_num) + 1, 1);
		dim3 kernel_get_neighbors_ids_thread(buffer_size_grained_thread_num, buffer_size_grained_thread_num);
		kernel_get_neighbors_ids<<<kernel_get_neighbors_ids_block, kernel_get_neighbors_ids_thread>>>(d_src_indexs, neighbors_sizes_list, edge_adjmatrix, 
																										cumulative_size_list,
																										neighbor_position_buffer, neighbor_id_buffer, buffer_size);
		cudaDeviceSynchronize();
		
		//get neighbors' indexs list
		//neighbor_index_buffer = storage_ -> GetDstAutoIndexing(neighbor_id_buffer, buffer_size);
		//get neighbors' weights list
		float* all_edge_weights = storage_ -> GetAllEdgeWeight();

		dim3 kernel_get_neighbors_weights_block(buffer_size / (buffer_size_grained_thread_num * buffer_size_grained_thread_num) + 1, 1);
		dim3 kernel_get_neighbors_weights_thread(buffer_size_grained_thread_num, buffer_size_grained_thread_num);
		kernel_get_neighbors_weights<<<kernel_get_neighbors_weights_block, kernel_get_neighbors_weights_thread>>>(neighbor_id_buffer, neighbor_weight_buffer,
																																	 all_edge_weights, buffer_size);
		cudaDeviceSynchronize();

		//calculate the partial sum and partial prefix sums

		float* partial_sums;
		cudaMallocManaged(&partial_sums, batch_size * sizeof(float));
		cudaMemset(partial_sums, 0, batch_size * sizeof(float));
		// dim3 kernel_get_partial_sums_block(batch_size / (batch_size_grained_thread_num * batch_size_grained_thread_num) + 1, 1);
		// dim3 kernel_get_partial_sums_thread(batch_size_grained_thread_num, batch_size_grained_thread_num);
		// kernel_get_partial_sums<<<kernel_get_partial_sums_block, kernel_get_partial_sums_thread>>>(neighbor_weight_buffer, partial_sums,
		// 																					 cumulative_size_list, batch_size, buffer_size);
		cudaDeviceSynchronize();
		for(int idx = 0; idx < batch_size; idx++){
			if (idx < batch_size - 1){
				partial_sums[idx] = thrust::reduce(neighbor_weight_buffer + cumulative_size_list[idx], neighbor_weight_buffer + cumulative_size_list[idx + 1]);
				thrust::exclusive_scan(thrust::device, neighbor_weight_buffer + cumulative_size_list[idx], neighbor_weight_buffer + cumulative_size_list[idx + 1], neighbor_weight_buffer + cumulative_size_list[idx]);
			}else if(idx == batch_size - 1){
				partial_sums[idx] = thrust::reduce(neighbor_weight_buffer + cumulative_size_list[idx], neighbor_weight_buffer + buffer_size);
				thrust::exclusive_scan(thrust::device, neighbor_weight_buffer + cumulative_size_list[idx], neighbor_weight_buffer + buffer_size, neighbor_weight_buffer + cumulative_size_list[idx]);
			}
		}
		cudaDeviceSynchronize();

		//get the cdf 
		dim3 kernel_get_cdf_block(buffer_size / (buffer_size_grained_thread_num * buffer_size_grained_thread_num) + 1, 1);
		dim3 kernel_get_cdf_thread(buffer_size_grained_thread_num, buffer_size_grained_thread_num);
		kernel_get_cdf<<<kernel_get_cdf_block, kernel_get_cdf_thread>>>(neighbor_weight_buffer, partial_sums, neighbor_position_buffer, buffer_size);
		cudaDeviceSynchronize();
		for(int i = 0; i < buffer_size; i++){
			std::cout<<"p "<<neighbor_position_buffer[i]<<" ";
			std::cout<<"w "<<neighbor_weight_buffer[i]<<" ";
			std::cout<<"i "<<neighbor_id_buffer[i]<<"\n";
		}

		//GPU kernel for sampling
		dim3 kernel_get_all_dst_ids_block(buffer_size / (buffer_size_grained_thread_num * buffer_size_grained_thread_num) + 1, 1);
		dim3 kernel_get_all_dst_ids_thread(buffer_size_grained_thread_num, buffer_size_grained_thread_num);

		//dst ids
		IdType* GPU_dst_edges_ids;
		//IdType* GPU_dst_nodes_ids;
		cudaMallocManaged(&GPU_dst_edges_ids, count * batch_size * sizeof(IdType));
		cudaMemset(GPU_dst_edges_ids, 0, count * batch_size * sizeof(IdType));
		//cudaMallocManaged(&GPU_dst_nodes_ids, count * batch_size * sizeof(IdType));
		//generate random numbers
		float* random_number_buffer;
		cudaMallocManaged(&random_number_buffer, count * batch_size * sizeof(float));
		cudaMemset(random_number_buffer, 0, count * batch_size * sizeof(float));
		dim3 kernel_generate_random_number_block(count * batch_size / (batch_size_grained_thread_num * batch_size_grained_thread_num) + 1, 1);
		dim3 kernel_generate_random_number_thread(batch_size_grained_thread_num, batch_size_grained_thread_num);
		kernel_generate_random_number<<<kernel_generate_random_number_block, kernel_generate_random_number_thread>>>(random_number_buffer, count * batch_size);
		cudaDeviceSynchronize();
		for (int i = 0; i < count * batch_size; i++){
			std::cout<<"r "<<random_number_buffer[i]<<"\n";
		}
		//sampling
		kernel_get_all_dst_ids<<<kernel_get_all_dst_ids_block, kernel_get_all_dst_ids_thread>>>(neighbor_position_buffer, neighbor_weight_buffer, neighbor_id_buffer,
																					random_number_buffer,
																					//GPU_dst_nodes_ids, 
																					GPU_dst_edges_ids,
																					count, batch_size, buffer_size);
		cudaDeviceSynchronize();
		//test sampling result
		for(int i = 0; i < count * batch_size; i++){
			std::cout<<GPU_dst_edges_ids[i]<<"\n";
		}
		//IndexType* h_result = (IndexType*)malloc(batch_size * sizeof(IndexType));
		//cudaMemcpy(h_result, d_result, batch_size * sizeof(IndexType), cudaMemcpyDefault);
		////////////////////////////////////////remember free ////////////////////////////////////////
		//free storage
		cudaFree(GPU_dst_edges_ids);
		//cudaFree(GPU_dst_nodes_ids);
		cudaFree(d_src_ids);
		cudaFree(d_src_indexs);
		cudaFree(neighbors_sizes_list);
		cudaFree(cumulative_size_list);
		cudaFree(neighbor_position_buffer);
		cudaFree(neighbor_weight_buffer);
		cudaFree(neighbor_id_buffer);
		//cudaFree(neighbor_index_buffer);
		cudaFree(partial_sums);
		cudaFree(random_number_buffer);
		
		//free(h_query);
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