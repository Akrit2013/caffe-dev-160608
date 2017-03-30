#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/grad_to_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Forward_gpu_kernel(const int nthreads,
    const Dtype* const bottom_data,
	Dtype* top_data,
	const int num,
	const int height,
	const int width){
  CUDA_KERNEL_LOOP(index, nthreads) {
	  const int n = index / height;
	  const int h = index % height;

	  const int stride = height * width;
	  const int bottom_idx = (n * 2 * height + h) * width;
	  const int top_idx = (n * 3 * height + h) * width;

	  for (int w = 0; w < width; w++){
		  const Dtype dx = bottom_data[bottom_idx+w];
		  const Dtype dy = bottom_data[bottom_idx+w+stride];

		  Dtype z = 1;
		  Dtype x = - dx * z;
		  Dtype y = - dy * z;

		  // Normalize the [x, y, z]
		  const Dtype norm = sqrt(x*x+y*y+z*z);
		  if (norm == 0){
			  x = 0;
			  y = 0;
			  z = 0;
		  }else{
			  x /= norm;
			  y /= norm;
			  z /= norm;
		  }

		  top_data[top_idx+w] = x;
		  top_data[top_idx+w+stride] = y;
		  top_data[top_idx+w+stride*2] = z;
	  }
  }
}

template<typename Dtype>
__global__ void Backward_gpu_kernel(
		const int nthreads,
		const Dtype* const top_diff,
		const Dtype* const top_data,
		Dtype* bottom_diff,
		const int num,
		const int height,
		const int width){
  CUDA_KERNEL_LOOP(index, nthreads) {
	  const int n = index / height;
	  const int h = index % height;

	  const int stride = height * width;
	  const int bottom_idx = (n * 2 * height + h) * width;
	  const int top_idx = (n * 3 * height + h) * width;

	  for (int w = 0; w < width; w++){
		  const Dtype z = top_data[top_idx + w + stride * 2];
		  const Dtype diff_dx = - top_diff[top_idx + w] / z;
		  const Dtype diff_dy = - top_diff[top_idx + w + stride] / z;

		  bottom_diff[bottom_idx] = diff_dx;
		  bottom_diff[bottom_idx + stride] = diff_dy;
	  }
  }
}

template <typename Dtype>
void GradToNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();

  const int num = bottom[0]->num();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();

  const int num_kernels = num * height;

  Forward_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
		  num_kernels, bottom_data, top_data, num, height, width);
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
void GradToNormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);

  const int num = bottom[0]->num();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();

  const int num_kernels = num * height;

  Backward_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
		  num_kernels, top_diff, top_data, bottom_diff, num, height, width);
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(GradToNormLayer);


}  // namespace caffe
