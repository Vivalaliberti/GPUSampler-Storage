#include <algorithm>
#include <functional>
#include <iostream>
#include "GPU_AdjacentMatrix.cuh"

class GPUMemoryAdjMatrix : public GPUAdjMatrix {
public:
	
	~GPUMemoryAdjMatrix() override {
		cudaFree(adj_nodes_);
		cudaFree(GPU_Matrix_cols_);
		cudaFree(GPU_Matrix_cols_capacity_);
	}
	
	//initialize the adjmatrix
	GPUMemoryAdjMatrix(GPUAutoIndexing* indexing){
		src_indexing_ = indexing;
		cudaMallocManaged(&adj_nodes_, init_row_cap * sizeof(IdType*));
		cudaMallocManaged(&adj_edges_, init_row_cap * sizeof(IdType*));
		//row
		GPU_Matrix_rows_ = 0;
		GPU_Matrix_rows_capacity_ = init_row_cap;
		//cols on CPU/GPU
		//CPU_Matrix_cols_ = (IdType*)malloc(init_rol_cap * sizeof(IdType));
		cudaMallocManaged(&GPU_Matrix_cols_, init_row_cap * sizeof(IdType));
		//CPU_Matrix_cols_capacity_ = (IdType*)malloc(init_rol_cap * sizeof(IdType));
		cudaMallocManaged(&GPU_Matrix_cols_capacity_, init_row_cap * sizeof(IdType));
		//initialize
		cudaMemset(GPU_Matrix_cols_, 0, GPU_Matrix_rows_capacity_ * sizeof(IdType));
		//cudaMemset(GPU_Matrix_cols_capacity_, init_col_cap, GPU_Matrix_rows_capacity_ * sizeof(IdType));
		dim3 kernel_init_block(GPU_Matrix_rows_capacity_/(kernel_init_thread_num * kernel_init_thread_num) + 1, 1);
		dim3 kernel_init_thread(kernel_init_thread_num, kernel_init_thread_num);
		Init_Device_Array<<<kernel_init_block, kernel_init_thread>>>(GPU_Matrix_cols_capacity_, init_col_cap, GPU_Matrix_rows_capacity_);
		cudaDeviceSynchronize();
		can_have_same_neighbor_ = true;
	}
	//return the row size
	IdType Row_Size() const override {
		return GPU_Matrix_rows_;
    }
    IdType Row_Cap() const override {
        return GPU_Matrix_rows_capacity_;
    }
	//return the col sizes
	IdType* Col_Size() const override {
		return GPU_Matrix_cols_;
    }
    IdType* Col_Cap() const override {
        return GPU_Matrix_cols_capacity_;
	}
	//void Can_Have_Same_neighbor(bool can_cannot) const {
	//	can_have_same_neighbor_ = can_cannot;
	//}
	//add node one by one
	void Add(IdType edge_id, IdType src_id, IdType dst_id) override {
		IdType* single_id_list;
		cudaMallocManaged(&single_id_list, sizeof(IdType));
		single_id_list[0] = src_id;
		IndexType* d_src_index;
		cudaMallocManaged(&d_src_index, sizeof(IndexType));
		d_src_index = src_indexing_ -> Get(single_id_list, 1); //to be tested

		IndexType src_index = d_src_index[0];
		if(src_index < GPU_Matrix_rows_){
			if(GPU_Matrix_cols_[src_index] + 1 >= GPU_Matrix_cols_capacity_[src_index] * LOAD_FACTOR){
				Expand_Cols(src_index);
			}
			if(can_have_same_neighbor_){
				adj_nodes_[src_index][GPU_Matrix_cols_[src_index]] = dst_id;
				adj_edges_[src_index][GPU_Matrix_cols_[src_index]] = edge_id;
				std::cout<<"new node: "<<src_index<<" "<<GPU_Matrix_cols_[src_index]<<" "<<adj_nodes_[src_index][GPU_Matrix_cols_[src_index]]<<std::endl; 
				//std::cout<<"new edge: "<<src_index<<" "<<GPU_Matrix_cols_[src_index]<<" "<<adj_edges_[src_index][GPU_Matrix_cols_[src_index]]<<endl; 
				GPU_Matrix_cols_[src_index] += 1;
			}else{
				int i;
            	for(i = 0; i < GPU_Matrix_cols_[src_index]; i++){
              	   	if(adj_nodes_[src_index][i] == dst_id || adj_edges_[src_index][i] == edge_id){
               	    	return;
                	}
                	if(adj_nodes_[src_index][i] == 0 && adj_edges_[src_index][i] == 0){
                	    break;
					}	
				}
				adj_nodes_[src_index][i] = dst_id;
				adj_edges_[src_index][i] = edge_id;
				std::cout<<"new node: "<<src_index<<" "<<i<<" "<<adj_nodes_[src_index][i]<<std::endl; 
				//std::cout<<"new edge: "<<src_index<<" "<<i<<" "<<adj_edges_[src_index][i]<<endl; 
				GPU_Matrix_cols_[src_index] += 1;
			}
		}else{
			if(src_index >= GPU_Matrix_rows_capacity_ * LOAD_FACTOR){
				Expand_Rows();
			}
			IdType* new_node_row;
			cudaMallocManaged(&new_node_row, init_col_cap * sizeof(IdType));
			cudaMemset(new_node_row, 0, init_col_cap * sizeof(IdType));
			IdType* new_edge_row;
			cudaMallocManaged(&new_edge_row, init_col_cap * sizeof(IdType));
			cudaMemset(new_edge_row, 0, init_col_cap * sizeof(IdType));
			adj_nodes_[src_index] = new_node_row;
			adj_edges_[src_index] = new_edge_row;
			//src_index will only be one larger than GPU_Matrix_rows_+1 according to the way graph are built
			GPU_Matrix_rows_ = GPU_Matrix_rows_ + 1;
			adj_nodes_[src_index][0] = dst_id;
			adj_edges_[src_index][0] = edge_id;

			std::cout<<"new node: "<<src_index<<" "<<0<<" "<<adj_nodes_[src_index][0]<<std::endl; 
			//std::cout<<"new edge: "<<src_index<<" "<<0<<" "<<adj_edges_[src_index][0]<<endl; 
			GPU_Matrix_cols_[src_index] += 1;
		}
	}

	//return the pointer of adjmatrix
	IdType** GetNodeAdjMatrix() const override {
		return adj_nodes_;
	}
	IdType** GetEdgeAdjMatrix() const override {
		return adj_edges_;
	}
	IdType* GetNeighbors(IndexType src_index) const override {
		return adj_nodes_[src_index];
	}
	IdType* GetOutEdges(IndexType src_index) const override {
		return adj_edges_[src_index];
	}
private:

	IdType** adj_nodes_; 
	IdType** adj_edges_;
	IdType* GPU_Matrix_cols_; 
	IdType* GPU_Matrix_cols_capacity_; 
	IdType GPU_Matrix_rows_;	
	IdType GPU_Matrix_rows_capacity_; 
	GPUAutoIndexing* src_indexing_;
	bool can_have_same_neighbor_;
	//IndexType* failed_index;
	
	//expand the row of adjmatrix
	void Expand_Rows(){
		//std::cout<<"expand row called\n";
		if(GPU_Matrix_rows_ + 1 >= GPU_Matrix_rows_capacity_ * LOAD_FACTOR){
			//initialize 
			IdType** new_node_adjmatrix;
			IdType** new_edge_adjmatrix;
			IdType* new_cols;
			IdType* new_cols_capacity;
			cudaMallocManaged(&new_cols, GPU_Matrix_rows_capacity_ * EXPAND_FACTOR * sizeof(IdType));
			cudaMallocManaged(&new_cols_capacity, GPU_Matrix_rows_capacity_ * EXPAND_FACTOR * sizeof(IdType));
			cudaMemset(new_cols, 0, GPU_Matrix_rows_capacity_ * EXPAND_FACTOR * sizeof(IdType));
			//cudaMemset(new_cols_capacity, init_col_cap, GPU_Matrix_rows_capacity_ * EXPAND_FACTOR * sizeof(IdType));
			dim3 kernel_init_block(GPU_Matrix_rows_capacity_ * EXPAND_FACTOR / (kernel_init_thread_num * kernel_init_thread_num) + 1, 1);
			dim3 kernel_init_thread(kernel_init_thread_num, kernel_init_thread_num);
			Init_Device_Array<<<kernel_init_block, kernel_init_thread>>>(new_cols_capacity, init_col_cap, GPU_Matrix_rows_capacity_ * EXPAND_FACTOR);		
			cudaMallocManaged(&new_node_adjmatrix, GPU_Matrix_rows_capacity_ * EXPAND_FACTOR * sizeof(IdType*));
			cudaMallocManaged(&new_edge_adjmatrix, GPU_Matrix_rows_capacity_ * EXPAND_FACTOR * sizeof(IdType*));

			//recover
			cudaMemcpy(new_cols, GPU_Matrix_cols_, GPU_Matrix_rows_ * sizeof(IdType), cudaMemcpyDefault);
			cudaMemcpy(new_cols_capacity, GPU_Matrix_cols_capacity_, GPU_Matrix_rows_ * sizeof(IdType),cudaMemcpyDefault);
			dim3 kernel_block(GPU_Matrix_rows_/(kernel_repoint_thread_num * kernel_repoint_thread_num) + 1, 1);
			dim3 kernel_thread(kernel_repoint_thread_num, kernel_repoint_thread_num);
			RePoint<<<kernel_block,kernel_thread>>>(new_node_adjmatrix, adj_nodes_, GPU_Matrix_rows_);
			RePoint<<<kernel_block,kernel_thread>>>(new_edge_adjmatrix, adj_edges_, GPU_Matrix_rows_);
			GPU_Matrix_rows_capacity_ = GPU_Matrix_rows_capacity_ * EXPAND_FACTOR;
			cudaFree(adj_nodes_);
			adj_nodes_ = new_node_adjmatrix;
			cudaFree(adj_edges_);
			adj_edges_ = new_edge_adjmatrix;
			cudaFree(GPU_Matrix_cols_);
			GPU_Matrix_cols_ = new_cols;
			cudaFree(GPU_Matrix_cols_capacity_);
			GPU_Matrix_cols_capacity_ = new_cols_capacity;
			cudaDeviceSynchronize();	
		}
	}
	//expand the src_index'th col of adjmatrix
	void Expand_Cols(IndexType src_index){
        //std::cout<<"expand cols called!\n";
		if(GPU_Matrix_cols_[src_index] + 1 >= GPU_Matrix_cols_capacity_[src_index] * LOAD_FACTOR){
			//initialize
			IdType* new_node_cols;
			IdType* new_edge_cols;
			cudaMallocManaged(&new_node_cols, GPU_Matrix_cols_capacity_[src_index] * EXPAND_FACTOR * sizeof(IdType));
			cudaMemset(new_node_cols, 0, GPU_Matrix_cols_capacity_[src_index] * EXPAND_FACTOR * sizeof(IdType));
			cudaMallocManaged(&new_edge_cols, GPU_Matrix_cols_capacity_[src_index] * EXPAND_FACTOR * sizeof(IdType));
			cudaMemset(new_edge_cols, 0, GPU_Matrix_cols_capacity_[src_index] * EXPAND_FACTOR * sizeof(IdType));
			//recover
			cudaMemcpy(new_node_cols, adj_nodes_[src_index], GPU_Matrix_cols_[src_index] * sizeof(IdType), cudaMemcpyDefault);
			cudaMemcpy(new_edge_cols, adj_edges_[src_index], GPU_Matrix_cols_[src_index] * sizeof(IdType), cudaMemcpyDefault);
			GPU_Matrix_cols_capacity_[src_index] *= EXPAND_FACTOR;
			cudaFree(adj_nodes_[src_index]);
			cudaFree(adj_edges_[src_index]);
			adj_nodes_[src_index] = new_node_cols;
			adj_edges_[src_index] = new_edge_cols;
			cudaDeviceSynchronize();
		}	
	}
	
};
	
	//repoint the adjmatrix to a new space
	__global__ void RePoint(IdType** new_matrix, IdType** old_matrix, 
													IdType GPU_Matrix_rows){
		int ix = blockDim.x * blockIdx.x + threadIdx.x;
		int iy = blockDim.y * blockIdx.y + threadIdx.y;
		int idx = ix + (gridDim.x * blockDim.x) * iy;											
		if(idx < GPU_Matrix_rows){
			new_matrix[idx] = old_matrix[idx];
		}
	}
	//initialize the array element by element(not by bytes)
	__global__ void Init_Device_Array(IdType* array, IdType init_value, IndexType batch_size){
		int ix = blockDim.x * blockIdx.x + threadIdx.x;
		int iy = blockDim.y * blockIdx.y + threadIdx.y;
		int idx = ix + (gridDim.x * blockDim.x) * iy;
		if(idx < batch_size){
			array[idx] = init_value;
		}
	}

GPUAdjMatrix* NewGPUMemoryAdjMatrix(GPUAutoIndexing* indexing){
	return new GPUMemoryAdjMatrix(indexing);
}
	/*
	int main(){
		GPUMemoryAdjMatrix matrix;
		IdType row;
		IdType** adj_nodes;

		for(int i = 0; i<15; i++){
			matrix.Add(0, i, i+1);
		}
		for(int j = 0; j<200; j++){
			matrix.Add(j, j, j);
		}
		adj_nodes = matrix.GetNodeAdjMatrix();
		//IdType* GPU_Matrix_cols_ = matrix.Col_Cap();
	   // IndexType src[10] = {1,2,3,1,2,4,5,6,7,8};
		//IdType dst[10] = {1,2,3,4,5,6,7,8,9,0};
		//IndexType* failed_index = matrix.Bulk_Add(src, dst, 10);
		row = matrix.Row_Size();
		std::cout<<"row: "<<row<<endl;
		std::cout<<"adj_nodes[0][0] "<<adj_nodes[0][0]<<endl;
	}*/
	//__global__ void kernel_self_mul(IdType* old, IdType factor){
	//		int idx = threadIdx.x + BlockDim.x * BlockIdx.x;
	//	old[idx] = old[idx] * factor;
	//}	
	/*
    __global__ void kernel_single_add(IdType** adj_nodes_, IndexType src_index, IdType* GPU_Matrix_cols_, IdType dst_id){
		int idx = threadIdx.x + blockDim.x * blockIdx.x;
		if(idx <= GPU_Matrix_cols_[src_index] && adj_nodes_[src_index][idx] != dst_id){
			if(adj_nodes_[src_index][idx] == 0){
				adj_nodes_[src_index][idx] = dst_id;
				GPU_Matrix_cols_[src_index] = GPU_Matrix_cols_[src_index] + 1;
			}//idx = GPU_Matrix_cols_[src_index]
		}
	}
	//assume the space is large enough
	__global__ void kernel_bulk_add(IdType** adj_nodes_, IndexType* src_indexs, IdType* GPU_Matrix_cols_, IdType* dst_ids, IndexType batch_size, IndexType* failed_index){
		int idx = threadIdx.x + blockDim.x * blockIdx.x;
		if(idx < batch_size){
			int col_num;
			for(col_num = 0; col_num < GPU_Matrix_cols_[idx]; col_num++){
				if(adj_nodes_[idx][col_num] == dst_ids[idx]){
					return;
				}
				if(adj_nodes_[idx][col_num] == 0){
					break;
				}
            }
            IdType check;
			check = atomicCAS((int*)(adj_nodes_[idx] + col_num), 0, (int)(dst_ids[idx]));
            if(check == 0){
                atomicAdd((int*)(GPU_Matrix_cols_ + idx), 1);
            }else{
                atomicAdd((int*)(failed_index + idx), 1);
            }

		}
    }
    //consider the conflic problem
    __global__ void Expand_Cols_bulk(IdType**adj_nodes_, IdType* GPU_Matrix_cols_, IdType* GPU_Matrix_cols_capacity_,
                                                                         IndexType* src_indexs, IndexType batch_size){
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if(idx < batch_size && ( GPU_Matrix_cols_[(src_indexs[idx])] + 1 >= GPU_Matrix_cols_capacity_[(src_indexs[idx])] * LOAD_FACTOR) ){
            IdType* new_col;
            new_col = (IdType*)malloc(GPU_Matrix_cols_capacity_[(src_indexs[idx])] * EXPAND_FACTOR * sizeof(IdType));
            //cudaMemcpyAsync(new_col, adj_nodes_[(src_indexs[idx])], GPU_Matrix_cols_[(src_indexs[idx])] * sizeof(IdType), cudaMemcpyDeviceToDevice);
            free(adj_nodes_[(src_indexs[idx])]);
            atomicExch((int*)(GPU_Matrix_cols_capacity_ + src_indexs[idx]), (int)(GPU_Matrix_cols_capacity_[(src_indexs[idx])] * EXPAND_FACTOR));
            //adj_nodes_[(src_indexs[idx])] = new_col;
            atomicExch((int*)(adj_nodes_ + src_indexs[idx]), (int)new_col);
        }
	}*/
    //add node bulk by bulk 
	/*
	IndexType* Bulk_Add(IndexType* src_indexs, IdType* dst_ids, IndexType batch_size){
		while(batch_size + GPU_Matrix_rows_ >= GPU_Matrix_rows_capacity_ * LOAD_FACTOR){
			Expand_Rows();
        }
        dim3 kernel_expand_block(batch_size/(kernel_expand_thread_num * kernel_expand_thread_num) + 1, 1);
        dim3 kernel_expand_thread(kernel_expand_thread_num, kernel_expand_thread_num);
        Expand_Cols_bulk<<<kernel_expand_block, kernel_expand_thread>>>(adj_nodes_, GPU_Matrix_cols_, GPU_Matrix_cols_capacity_, src_indexs, batch_size);
        //ensured enough space

        cudaMallocManaged(&failed_index, batch_size * sizeof(IdType));
		dim3 kernel_add_block(batch_size/(kernel_add_thread_num * kernel_add_thread_num) + 1, 1);
		dim3 kernel_add_thread(kernel_add_thread_num, kernel_add_thread_num);
		kernel_bulk_add<<<kernel_add_block, kernel_add_thread>>>(adj_nodes_, src_indexs, GPU_Matrix_cols_, dst_ids, batch_size, failed_index);
		return failed_index;
	}
	*/



