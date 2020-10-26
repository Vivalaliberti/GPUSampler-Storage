#include "GPUAdjacentMatrix.cuh"

class GPUMemoryAdjMatrix : public GPUAdjMatrix {
public:
	~GPUMemoryAdjMatrix() override {
		cudaFree(adj_nodes_);
		cudaFree(GPU_Matrix_cols_);
		cudaFree(GPU_Matrix_cols_capacity_);
	}
	//initialize the adjmatrix
	GPUMemoryAdjMatrix(){
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
	void Add(IndexType src_index, IdType dst_id, IdType edge_id) override {
		if(src_index < GPU_Matrix_rows_){
			if(GPU_Matrix_cols_[src_index] + 1 >= GPU_Matrix_cols_capacity_[src_index] * LOAD_FACTOR){
				Expand_Cols(src_index);
			}
			if(can_have_same_neighbor_){
				adj_nodes_[src_index][GPU_Matrix_cols_[src_index]] = dst_id;
				adj_edges_[src_index][GPU_Matrix_cols_[src_index]] = edge_id;
				//cout<<"new node: "<<src_index<<" "<<GPU_Matrix_cols_[src_index]<<" "<<adj_nodes_[src_index][GPU_Matrix_cols_[src_index]]<<endl; 
				//cout<<"new edge: "<<src_index<<" "<<GPU_Matrix_cols_[src_index]<<" "<<adj_edges_[src_index][GPU_Matrix_cols_[src_index]]<<endl; 
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
				//cout<<"new node: "<<src_index<<" "<<i<<" "<<adj_nodes_[src_index][i]<<endl; 
				//cout<<"new edge: "<<src_index<<" "<<i<<" "<<adj_edges_[src_index][i]<<endl; 
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
			//cout<<"new node: "<<src_index<<" "<<0<<" "<<adj_nodes_[src_index][0]<<endl; 
			//cout<<"new edge: "<<src_index<<" "<<0<<" "<<adj_edges_[src_index][0]<<endl; 
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
	bool can_have_same_neighbor_;
	//IndexType* failed_index;
	
	//expand the row of adjmatrix
	void Expand_Rows(){
		//cout<<"expand row called\n";
		if(GPU_Matrix_rows_ + 1 >= GPU_Matrix_rows_capacity_ * LOAD_FACTOR){
			//initialize 
			IdType** new_node_adjmatrix;
			IdType** new_edge_adjmatrix;
			IdType* new_cols;
			IdType* new_cols_capacity;
			cudaMallocManaged(&new_cols, GPU_Matrix_rows_capacity_ * expand_factor * sizeof(IdType));
			cudaMallocManaged(&new_cols_capacity, GPU_Matrix_rows_capacity_ * expand_factor * sizeof(IdType));
			cudaMemset(new_cols, 0, GPU_Matrix_rows_capacity_ * expand_factor * sizeof(IdType));
			//cudaMemset(new_cols_capacity, init_col_cap, GPU_Matrix_rows_capacity_ * expand_factor * sizeof(IdType));
			dim3 kernel_init_block(GPU_Matrix_rows_capacity_ * expand_factor / (kernel_init_thread_num * kernel_init_thread_num) + 1, 1);
			dim3 kernel_init_thread(kernel_init_thread_num, kernel_init_thread_num);
			Init_Device_Array<<<kernel_init_block, kernel_init_thread>>>(new_cols_capacity, init_col_cap, GPU_Matrix_rows_capacity_ * expand_factor);		
			cudaMallocManaged(&new_node_adjmatrix, GPU_Matrix_rows_capacity_ * expand_factor * sizeof(IdType*));
			cudaMallocManaged(&new_edge_adjmatrix, GPU_Matrix_rows_capacity_ * expand_factor * sizeof(IdType*));

			//recover
			cudaMemcpy(new_cols, GPU_Matrix_cols_, GPU_Matrix_rows_ * sizeof(IdType), cudaMemcpyDefault);
			cudaMemcpy(new_cols_capacity, GPU_Matrix_cols_capacity_, GPU_Matrix_rows_ * sizeof(IdType),cudaMemcpyDefault);
			dim3 kernel_block(GPU_Matrix_rows_/(kernel_repoint_thread_num * kernel_repoint_thread_num) + 1, 1);
			dim3 kernel_thread(kernel_repoint_thread_num, kernel_repoint_thread_num);
			RePoint<<<kernel_block,kernel_thread>>>(new_node_adjmatrix, adj_nodes_, GPU_Matrix_rows_);
			RePoint<<<kernel_block,kernel_thread>>>(new_edge_adjmatrix, adj_edges_, GPU_Matrix_rows_);
			GPU_Matrix_rows_capacity_ = GPU_Matrix_rows_capacity_ * expand_factor;
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
        //cout<<"expand cols called!\n";
		if(GPU_Matrix_cols_[src_index] + 1 >= GPU_Matrix_cols_capacity_[src_index] * LOAD_FACTOR){
			//initialize
			IdType* new_node_cols;
			IdType* new_edge_cols;
			cudaMallocManaged(&new_node_cols, GPU_Matrix_cols_capacity_[src_index] * expand_factor * sizeof(IdType));
			cudaMemset(new_node_cols, 0, GPU_Matrix_cols_capacity_[src_index] * expand_factor * sizeof(IdType));
			cudaMallocManaged(&new_edge_cols, GPU_Matrix_cols_capacity_[src_index] * expand_factor * sizeof(IdType));
			cudaMemset(new_edge_cols, 0, GPU_Matrix_cols_capacity_[src_index] * expand_factor * sizeof(IdType));
			//recover
			cudaMemcpy(new_node_cols, adj_nodes_[src_index], GPU_Matrix_cols_[src_index] * sizeof(IdType), cudaMemcpyDefault);
			cudaMemcpy(new_edge_cols, adj_edges_[src_index], GPU_Matrix_cols_[src_index] * sizeof(IdType), cudaMemcpyDefault);
			GPU_Matrix_cols_capacity_[src_index] *= expand_factor;
			cudaFree(adj_nodes_[src_index]);
			cudaFree(adj_edges_[src_index]);
			adj_nodes_[src_index] = new_edge_cols;
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