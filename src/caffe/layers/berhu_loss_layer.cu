#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/berhu_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Forward_gpu_kernel(
		 const int nthreads,
		 const Dtype* const data_label,
		 Dtype* data_diff,
		 Dtype* bad_pixel_data,
		 const int num,
		 const int channels,
		 const int height,
		 const int width,
		 const bool has_max_label,
		 const bool has_min_label,
		 const bool has_invalid_label,
		 const Dtype max_label,
		 const Dtype min_label,
		 const Dtype invalid_label,
		 const Dtype C,
		 const bool has_h_rate,
		 const Dtype H){
	CUDA_KERNEL_LOOP(index, nthreads){
		const int n = index / height;
		const int h = index % height;

		const int data_offset = (n*channels*height+h)*width;
		const int bad_pixel_idx = index;
		const int interval = height * width;

		// Iter the width and channels
		for (int w = 0; w < width; w++){
			// Iter the channels
			int err_counter = 0;
			for (int c = 0; c < channels; c++){
				const int idx = data_offset + c * interval + w;
				Dtype dataval = data_label[idx];
				Dtype diffval = data_diff[idx];

				if (has_max_label && dataval > max_label){
					err_counter++;
				}else if(has_min_label && dataval < min_label){
					err_counter++;
				}else if(has_invalid_label && fabs(dataval - invalid_label) < 0.0001){
					err_counter++;
				}
				// alter the diff value
				if (diffval > 0 && diffval < C){
					// L1
					data_diff[idx] = C;
				}else if(diffval < 0 && -diffval < C){
					data_diff[idx] = -C;
				}
				if (has_h_rate && diffval > H){
					data_diff[idx] = H;
				}else if(has_h_rate && -diffval > H){
					data_diff[idx] = -H;
				}
			}

			// Only if all channels invalid, the pixel will be considered
			// as invalid
			if(err_counter == channels){
				bad_pixel_data[bad_pixel_idx] += channels;
				for (int c = 0; c < channels; c++){
					const int idx = data_offset + c * interval + w;
					data_diff[idx] = 0;
				}
			}
		}
	}
}

template <typename Dtype>
void BerhuLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());

  Dtype max_diff = 0;

  switch(c_rate_mode_){
	  case MAX:
		  // Get the abs max diff to determine the C
		  max_diff = caffe_gpu_amax(count, diff_.gpu_data(), 1);
		  // Calc the Threshold C
		  break;
	  case AVE:
		  // Calc the mean of the abs diff
		  caffe_gpu_asum(count, diff_.gpu_data(), &max_diff);
		  max_diff /= count;
		  break;
	  default:
		  LOG(FATAL) << "False c_rate_mode";
		  break;
  }
  Dtype C = fabs(max_diff * c_rate_);
  Dtype H = fabs(max_diff * h_rate_);

  Dtype* data_diff = diff_.mutable_gpu_data();
  // const Dtype* data_pred = bottom[0]->cpu_data();
  const Dtype* data_label = bottom[1]->gpu_data();
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  // The number of kernel is num * height, process a row each time
  const int num_kernels = num * height;
  // Set the bad_pixel_ buffer to zero
  Dtype* bad_pixel_data = bad_pixel_.mutable_gpu_data();
  caffe_gpu_set(bad_pixel_.count(), Dtype(0), bad_pixel_data);
  // Find the bad pixel and alter the diff
  Forward_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
		 num_kernels,
		 data_label,
		 data_diff,
		 bad_pixel_data,
		 num,
		 channels,
		 height,
		 width,
		 has_max_label_,
		 has_min_label_,
		 has_invalid_label_,
		 max_label_,
		 min_label_,
		 invalid_label_,
		 C,
		 has_h_rate_,
		 H);

  Dtype bad_pixel_count;
  caffe_gpu_asum(bad_pixel_.count(), bad_pixel_data, &bad_pixel_count);
  Dtype dot;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  Dtype loss = dot / Dtype(2) / (count-bad_pixel_count);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void BerhuLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      Dtype alpha;
	  if (normalize_){
		  alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->count();
	  }else{
		  alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
	  }
      caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(BerhuLossLayer);

}  // namespace caffe
