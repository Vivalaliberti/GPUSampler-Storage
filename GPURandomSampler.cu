#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/linear_congruential_engine.h>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include "GPUMemoryAdjacentMatrix.cuh"
#include "GPUAuto_Indexing.cuh"
#include "slab_hash.cuh"
#define kernel_sampling_thread_num 8

using namespace std;

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

__global__ void kernel_random_sampler(IndexType* GPU_src_index, IdType* GPU_all_neighbor_ids[], IdType* GPU_all_edge_ids[], IdType* GPU_col_size,
										IndexType batch_size, IndexType count, 
								IdType* GPU_dst_nodes_ids, IdType* GPU_dst_edges_ids){
	int ix = blockDim.x * blockIdx.x + threadIdx.x;
	int iy = blockDim.y * blockIdx.y + threadIdx.y;
	int idx = ix + (gridDim.x * blockDim.x) * iy;
	thrust::minstd_rand engine;
	if (idx<batch_size){
		IdType GPU_src_index_ = GPU_src_index[idx];
		IdType* GPU_neighbor_ids = GPU_all_neighbor_ids[GPU_src_index_];
		IdType* GPU_edge_ids = GPU_all_edge_ids[GPU_src_index_];
		IdType col_size = GPU_col_size[GPU_src_index_];
		thrust::uniform_int_distribution<> dist(0, col_size - 1);
		
		for (IndexType j = 0; j < count; ++j) {
	    	IndexType dst_index = dist(engine);
			GPU_dst_nodes_ids[count * idx + j] = GPU_neighbor_ids[dst_index];
			GPU_dst_edges_ids[count * idx + j] = GPU_edge_ids[dst_index];
        }
	}
}

class GPURandomSampler /* : public Sampler  */{
public:
	virtual ~GPURandomSampler() {}

	void IndexingBuild(GPUAutoIndexing* indexing){
		indexing_ = indexing;
	}
	void StorageBuild(GPUMemoryAdjMatrix* storage){
		storage_ = storage;
	}

  /*   Status Sample( const SamplingRequest* req,   SamplingResponse* res ) override { */
    void Sample(SamplingRequest* req) {			
		//settings
		IndexType count = req->NeighborCount();
		IndexType batch_size = req->Batch_size();	
		IdType* h_query = req->src_ids();
		IndexType num_queries = batch_size;
	
		IdType* d_query;
		cudaMalloc(&d_query, num_queries * sizeof(IdType));
		IndexType* d_result;
		cudaMalloc(&d_result, num_queries * sizeof(IndexType));
	
		cudaMemcpy(d_query, h_query, num_queries * sizeof(IdType), cudaMemcpyDefault);
		
		d_result = indexing_->Get(d_query, num_queries);
		
		IdType** GPU_ALL_EDGE_IDS = storage_->GetNodeAdjMatrix();
		IdType** GPU_ALL_NEIGHBOR_IDS = storage_->GetEdgeAdjMatrix();
		IdType* GPU_COl_SIZE = storage_->Col_Size();
	
		//GPU kernel for sampling
		IndexType block_num;
		block_num = batch_size/(kernel_sampling_thread_num * kernel_sampling_thread_num)+1;
		dim3 kernel_sampling_block(block_num,1);
		dim3 kernel_sampling_thread(kernel_sampling_thread_num, kernel_sampling_thread_num);
		cudaSetDevice(0);
		//dst ids
		IdType* GPU_dst_edges_ids;
		IdType* GPU_dst_nodes_ids;
		cudaMalloc(&GPU_dst_edges_ids, count * batch_size * sizeof(IdType));
		cudaMalloc(&GPU_dst_nodes_ids, count * batch_size * sizeof(IdType));
		//sampling
		kernel_random_sampler<<<kernel_sampling_block,kernel_sampling_thread>>>(d_result, GPU_ALL_NEIGHBOR_IDS, GPU_ALL_EDGE_IDS, GPU_COl_SIZE, batch_size, count,  GPU_dst_nodes_ids, GPU_dst_edges_ids);
		//test sampling result
		IdType* dst_nodes_ids;
		IdType* dst_edges_ids;
		dst_nodes_ids = (IdType*)malloc(count * batch_size * sizeof(IdType));
		dst_edges_ids = (IdType*)malloc(count * batch_size * sizeof(IdType));
		cudaMemcpy(dst_edges_ids, GPU_dst_edges_ids, count * batch_size * sizeof(IdType), cudaMemcpyDefault);
		cudaMemcpy(dst_nodes_ids, GPU_dst_nodes_ids, count * batch_size * sizeof(IdType), cudaMemcpyDefault);
	
		//sampled node/edges
		int i;
		for (i = 0; i < batch_size * count; i++) {
			if(dst_nodes_ids[i] > (2*(i/count) + 1)||dst_edges_ids[i] < (i/count)){
				break;
			}
			if(i%count == 0){
				cout<<"dst_id: "<<dst_nodes_ids[i]<<endl;
			}
		}
		if(i == batch_size * count){
			cout<<"sampling success\n";
		}
		//IndexType* h_result = (IndexType*)malloc(num_queries * sizeof(IndexType));
		//cudaMemcpy(h_result, d_result, num_queries * sizeof(IndexType), cudaMemcpyDefault);

		//free storage
		cudaFree(GPU_dst_edges_ids);
		cudaFree(GPU_dst_nodes_ids);
		cudaFree(d_result);
		free(h_query);
		free(dst_edges_ids);
		free(dst_nodes_ids);
		//return Status::OK();
  }

private:
	GPUMemoryAdjMatrix* storage_;
	GPUAutoIndexing* indexing_;
};

int main(){

	IndexType num_keys = 2<<10;
	IndexType num_queries = 2<<6;
	IndexType num_buckets = 2<<9;
	IdType* h_key = (IdType*)malloc(num_keys * sizeof(IdType));
	IndexType* h_value = (IndexType*)malloc(num_keys * sizeof(IndexType));
	IdType* h_query = (IdType*)malloc(num_queries * sizeof(IdType));
	IndexType* h_test = (IndexType*)malloc(num_queries * sizeof(IndexType));
	
	GPUMemoryAdjMatrix* storage = new GPUMemoryAdjMatrix();

	for(int i = 0; i < num_keys; i++){
		h_key[i] = i;
		h_value[i] = i;
		for(int k= 0; k < i + 2; k++){
			storage->Add(h_value[i], h_value[i] + k, h_value[i] + k);
		}
	}
	cout<<"add success\n";
	for(int j = 0; j < num_queries; j++){
		h_query[j] = j;
	}

	SamplingRequest* req = new SamplingRequest();
	req->Src_Ids = h_query;
	req->batch_size = num_queries;
	req->count = 5;

	GPUAutoIndexing* indexing = new GPUAutoIndexing();
	indexing->buildBulk(h_key, h_value, num_keys, num_buckets);
	cout<<"hash success\n";

	GPURandomSampler sampler;
	sampler.IndexingBuild(indexing);
	sampler.StorageBuild(storage);
	cout<<"sampler build success\n";

	sampler.Sample(req);

	return 0;
}

