#include <vector>

#include "caffe/layers/conv1x1_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void normalize_param_kernel(const int n, Dtype* weight_data,
		const int num, const int channels, const int height, const int width,
		const bool is_norm, const bool is_pos, const Dtype norm_val){
	CUDA_KERNEL_LOOP(index, n) {
		const int n_idx = index / height;
		const int h = index % height;
		// Loop the row
		for (int w = 0; w < width; w++){
			Dtype c_sum = Dtype(0);
			for (int c = 0; c < channels; c++){
				int index = ((n_idx*channels+c)*height+h)*width+w;
				Dtype val = weight_data[index];
				if (is_pos){
					weight_data[index] = Dtype((val>0) ? val:0);
				}
				c_sum += weight_data[index];
			}
			if (is_norm){
				Dtype rate = norm_val / c_sum;
				for(int c = 0; c < channels; c++){
					int index = ((n_idx*channels+c)*height+h)*width+w;
					weight_data[index] = rate*weight_data[index];
				}
			}
		}
	}
}

/*
template <typename Dtype>
void Conv1x1Layer<Dtype>::normalize_param_gpu(void){
	if (this->normalize_ == false && this->positive_ == false){
		return;
	}
	Dtype* weight_data = this->blobs_[0]->mutable_gpu_data();
	int num = this->blobs_[0]->num();
	int channels = this->blobs_[0]->channels();
	int height = this->blobs_[0]->height();
	int width = this->blobs_[0]->width();
	// Use a individual thread for each row
	int num_kernels = num * height;
	normalize_param_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
									CAFFE_CUDA_NUM_THREADS>>>(
		num_kernels, weight_data, num, channels, height, width, this->normalize_,
		this->positive_, this->norm_val_);
	CUDA_POST_KERNEL_CHECK;
}
*/

template <typename Dtype>
void Conv1x1Layer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  // Before each forward, regulate the weight
  // If test phase do nothing, only alter the value in the train phase
  if (this->phase_ == TRAIN && (this->normalize_ == true || this->positive_ == true)){
	  //normalize_param_gpu();
	  Dtype* weight_data = this->blobs_[0]->mutable_gpu_data();
	  int num = this->blobs_[0]->num();
	  int channels = this->blobs_[0]->channels();
	  int height = this->blobs_[0]->height();
	  int width = this->blobs_[0]->width();
	  // Use a individual thread for each row
	  int num_kernels = num * height;
	  normalize_param_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
		  CAFFE_CUDA_NUM_THREADS>>>(
				  num_kernels, weight_data, num, channels, height, width, this->normalize_,
				  this->positive_, this->norm_val_);
	  CUDA_POST_KERNEL_CHECK;
  }
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void Conv1x1Layer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(Conv1x1Layer);

}  // namespace caffe
