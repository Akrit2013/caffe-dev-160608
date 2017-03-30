#include "caffe/util/mytools.hpp"


namespace caffe {


template <typename Dtype>
__global__ void bilinear_interpolation_kernel(
		const int nthreads,
		const Dtype* const src_data,
		Dtype* dst_data,
		const int channels,
		const int height,
		const int width,
		const int new_height,
		const int new_width){
	CUDA_KERNEL_LOOP(index, nthreads){
		const int h = index / new_width;
		const int w = index % new_width;
		const int new_stride = new_height * new_width;
		const int stride = height * width; 

		Dtype h_rate = Dtype(height) / new_height;
		Dtype w_rate = Dtype(width) / width;
		// Find the position in the original image
		Dtype h_pos = h * h_rate;
        Dtype w_pos = w * w_rate;
		// Calc the reference pos of the surround 4 pixel
		int h0 = int(h_pos);
		int w0 = int(w_pos);
		Dtype u = h_pos - h0;
		Dtype v = w_pos - w0;
		if (h0 == height - 1){
			h0--;
			u = 1;
		}
		if (w0 == width - 1){
			w0--;
			v = 1;
		}
		// Iter the channels
		for (int c = 0; c < channels; c++){
			int offset = c * stride;
			Dtype val0 = src_data[offset + h0 * width + w0];
			Dtype val1 = src_data[offset + h0 * width + w0 + 1];
			Dtype val2 = src_data[offset + (h0 + 1) * width + w0];
			Dtype val3 = src_data[offset + (h0 + 1) * width + w0 + 1];

			int dst_idx = c * new_stride + h * new_width + w;
			dst_data[dst_idx] = (1-u)*(1-v)*val0 + (1-u)*v*val1 + u*(1-v)*val2 + u*v*val3;
		}
	
	}
}

template <typename Dtype>
void cuBilinearInterpolation(const Dtype* const src_data, Dtype* dst_data,
	   const int channels, const int height, const int width,
	   const int new_height, const int new_width){
	// Check the params
	CHECK_NE(src_data, dst_data) << "Can not take in-place operation";
	CHECK_GT(new_height, height) << "Currently can only perform zoom out";
	CHECK_GT(new_width, width) << "Currently can only perform zoom out";

	// The kernel num
	const int num_kernels = new_height * new_width;
	// Perform the interpolation
	bilinear_interpolation_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
			CAFFE_CUDA_NUM_THREADS>>>(num_kernels, src_data, dst_data,
					channels, height, width, new_height, new_width);
	CUDA_POST_KERNEL_CHECK;
	
}

template <typename Dtype>
__global__ void crop_and_resize_back_kernel(
		const int nthreads,
		const Dtype* const src_data,
		Dtype* dst_data,
	    const Dtype* const h_size_rate_data, 
		const Dtype* const w_size_rate_data,
	    const Dtype* const h_offset_rate_data,
		const Dtype* const w_offset_rate_data,
		const int num,
	    const int channels,
	    const int height,
	    const int width){
	CUDA_KERNEL_LOOP(index, nthreads){
		const int n = index / channels / height;
		const int c = (index / height) % channels;
		const int h = index % height;

		const Dtype h_rate = h_size_rate_data[n];
		const Dtype w_rate = w_size_rate_data[n];
		const Dtype h_offset_rate = h_offset_rate_data[n];
		const Dtype w_offset_rate = w_offset_rate_data[n];

		const int crop_height = h_rate * height + 0.5;
		const int crop_width = w_rate * width + 0.5;

		const int h_offset = h_offset_rate * (height - crop_height) + 0.5;
		const int w_offset = w_offset_rate * (width - crop_width) + 0.5;

		int idx_offset = (n*channels+c)*height*width;

		// Calc the h pos of the reference pixel
		Dtype h_pos = Dtype(h) * h_rate;
        int h0 = int(h_pos);
		Dtype u = h_pos - h0;
		if (h0 == crop_height - 1){
			h0--;
			u = 1;
		}
		// Iter the width
		for(int w = 0; w < width; w++){
			Dtype w_pos = Dtype(w) * w_rate;
			int w0 = int(w_pos);
		    Dtype v = w_pos - w0;
			if (w0 == crop_width - 1){
				w0--;
				v = 1;
			}
			Dtype val0 = src_data[idx_offset + (h_offset+h0) * width + w_offset+w0];
			Dtype val1 = src_data[idx_offset + (h_offset+h0) * width + w_offset+w0 + 1];
			Dtype val2 = src_data[idx_offset + (h_offset+h0 + 1) * width + w_offset+ w0];
			Dtype val3 = src_data[idx_offset + (h_offset+h0 + 1) * width + w_offset+w0 + 1];

			dst_data[idx_offset+h*width+w] = (1-u)*(1-v)*val0 + (1-u)*v*val1 + u*(1-v)*val2 + u*v*val3;
		}

	}
}

template <typename Dtype>
void cuCropAndResizeBack(const Dtype* const src_data, Dtype* dst_data,
		const Dtype* const h_size_rate_data, const Dtype* const w_size_rate_data,
		const Dtype* const h_offset_rate_data, const Dtype* const w_offset_rate_data,
		const int num, const int channels, const int height, const int width){
	// Check the params
	CHECK_NE(src_data, dst_data) << "Can not take in-place operation";

	const int num_kernels = num * channels * height;

	crop_and_resize_back_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
		CAFFE_CUDA_NUM_THREADS>>>(num_kernels, src_data, dst_data,
				h_size_rate_data, w_size_rate_data, h_offset_rate_data, w_offset_rate_data, num, channels, height, width);
	CUDA_POST_KERNEL_CHECK;
}



template void cuBilinearInterpolation<float>(const float* const src_data, float* dst_data, const int channels, const int height, const int width, const int new_height, const int new_width);

template void cuBilinearInterpolation<double>(const double* const src_data, double* dst_data, const int channels, const int height, const int width, const int new_height, const int new_width);

template void cuCropAndResizeBack<float>(const float* const src_data, float* dst_data, const float* const h_size_rate_data, const float* const w_size_rate_data, const float* const h_offset_rate_data, const float* const w_offset_rate_data, const int num, const int channels, const int height, const int width);

template void cuCropAndResizeBack<double>(const double* const src_data, double* dst_data, const double* const h_size_rate_data, const double* const w_size_rate_data, const double* const h_offset_rate_data, const double* const w_offset_rate_data, const int num, const int channels, const int height, const int width);



}		// namespace caffe
