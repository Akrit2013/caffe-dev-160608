#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/depth_to_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Forward_gpu_kernel(const int nthreads,
    const Dtype* const bottom_data,
	Dtype* top_data,
	const int radius,
	const Dtype focal,
	const int num,
	const int height,
	const int width){
  CUDA_KERNEL_LOOP(index, nthreads) {
	  const int n = index / height;
	  const int h = index % height;

	  const int stride = height * width;
	  const int bottom_idx = (n * 1 * height + h) * width;
	  const int top_idx = (n * 3 * height + h) * width;

	  const int hu = max(0, h - radius);
	  const int hd = min(height - 1, h + radius);
	  const Dtype hs = hd - hu;

	  const int bottom_idx_hu = (n * height + hu) * width;
	  const int bottom_idx_hd = (n * height + hd) * width;

	  for (int w = 0; w < width; w++){
		  const int wl = max(0, w - radius);
		  const int wr = min(width - 1, w + radius);
		  const Dtype ws = wr - wl;
		  Dtype x = (bottom_data[bottom_idx + wl] - bottom_data[bottom_idx + wr]) / ws;
		  Dtype y = (bottom_data[bottom_idx_hu + w] - bottom_data[bottom_idx_hd + w]) / hs;
		  Dtype z = 1.0 / focal;

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

template <typename Dtype>
void DepthToNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();

  const int num = bottom[0]->num();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();

  const int num_kernels = num * height;

  Forward_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
		  num_kernels, bottom_data, top_data, radius, focal, num, height, width);
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(DepthToNormLayer);


}  // namespace caffe
