#include "caffe/layers/synctrans_fast_layer.hpp"
#include "caffe/util/mytools.hpp"


namespace caffe {

template <typename Dtype>
__global__ void forward_gpu_kernel(
					const int n,
				   	const Dtype* const bottom_data,
					const Dtype* const h_offset_data,
					const Dtype* const w_offset_data,
					const Dtype* const color_data,
					const Dtype* const mirror_data,
					const int num, 
					const int channels, 
					const int height, 
					const int width, 
					const int top_height,
					const int top_width,
					const int crop_height,
					const int crop_width,
					const bool mirror,
					const bool alter_color,
					const bool is_norm_map,
					const bool is_superpixel_map,
					const bool is_pairwise_map,
					const bool is_grad_map,
					Dtype* top_data){
	CUDA_KERNEL_LOOP(index, n){
		const int n_idx = index / height / channels;
		const int c = (index / height) % channels;
		const int h = index % height;

		// Calc the crop offset
		int h_start_idx = 0;
		int h_end_idx = height;

		int w_start_idx = 0;
		int w_end_idx = width;

		// Determine if perform mirror
		bool do_mirror = false;
		if (mirror){
			do_mirror = mirror_data[n_idx] > 0.5 ? true: false;
		}

		// If the current blob is superpixel_map_
		if (is_superpixel_map == true){
			// TODO: Not perform crop when use superpixel_map_, the top_width must
			// equal to the width
			// The top can only have one channel
			const int top_idx = (n_idx * top_height + h) * top_width;
			int bottom_idx = 0;
			if (do_mirror){
				// Copy the 2nd channel of bottom to the top
				bottom_idx = ((n_idx * channels + 1) * height + h) * width;
			}else{
				// Simply copy the 1st channel
				bottom_idx = ((n_idx * channels + 0) * height + h) * width;
			}
			// Direct copy the bottom data to top
			for (int w = 0; w < top_width; w++){
				top_data[top_idx+w] = bottom_data[bottom_idx+w];
			}
			return;
		}

		if (is_pairwise_map == true){
			// TODO: Can not perform the crop on the pairwise map
			// If mirror, take the upper trangle
			// If not mirror, take the lower trangle
			const int offset = (n_idx * channels + c) * top_height * width;
			if(do_mirror){
				for(int w = 0; w < h; w++){
					top_data[offset+h*width+w] = bottom_data[offset+w*width+h];
				}
				for(int w = h; w < width; w++){
					top_data[offset+h*width+w] = bottom_data[offset+h*width+w];
				}
			}else{
				for(int w = 0; w < h; w++){
					top_data[offset+h*width+w] = bottom_data[offset+h*width+w];
				}
				for(int w = h; w < width; w++){
					top_data[offset+h*width+w] = bottom_data[offset+w*width+h];
				}
			}

			return;
		}


		if (crop_height != 0){
			Dtype offset_rate = h_offset_data[n_idx];
			int margin = height - crop_height;
			int offset = offset_rate * margin + 0.5;
			h_start_idx = offset;
			h_end_idx = h_start_idx + crop_height;
		}

		if (crop_width != 0){
			Dtype offset_rate = w_offset_data[n_idx];
			int margin = width - crop_width;
			int offset = offset_rate * margin + 0.5;
			w_start_idx = offset;
			w_end_idx = w_start_idx + crop_width;
		}

		if (h_start_idx > h || h_end_idx < h+1) {
			return;
		}

		// Determine the color offset
		Dtype color_multi = 1;
		if (alter_color){
			color_multi += color_data[n_idx*channels+c];
		}

		const int bottom_idx = ((n_idx*channels+c)*height+h)*width;
		const int top_idx = ((n_idx*channels+c)*top_height+h-h_start_idx)*top_width;

		// Normal map change the direction of the X axis
		Dtype sign = 1;
		if ((is_grad_map || is_norm_map) && c == 0 && do_mirror){
			sign = -1;
		}

		// Iter the width
		if (do_mirror){
			for (int w = w_start_idx; w < w_end_idx; w++){
				top_data[top_idx+top_width-1-w+w_start_idx] = sign * color_multi*bottom_data[bottom_idx+w];
			}
		}else{
			for (int w = w_start_idx; w < w_end_idx; w++){
				top_data[top_idx+w-w_start_idx] = color_multi*bottom_data[bottom_idx+w];
			}
		}
	}
}


template <typename Dtype>
void SyncTransFastLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	// Deside the random parameter for the transform
	random_params();

	// Start to transform the data for each blobs
	const int blobs_num = bottom.size();

	const Dtype* h_offset_data = batch_crop_h_offset_.gpu_data();
	const Dtype* w_offset_data = batch_crop_w_offset_.gpu_data();
	const Dtype* color_data = batch_color_offset_.gpu_data();
	const Dtype* mirror_data = batch_mirror_.gpu_data();
	// The temp blobs used in the random crop
	Blob<Dtype> tmp_blobs;

	for (int i = 0; i < blobs_num; i++){
		const Dtype* bottom_data = bottom[i]->gpu_data();
		Dtype* out_data = NULL;
		if (crop_rate_ != 1){
			tmp_blobs.ReshapeLike(*top[i]);
			out_data = tmp_blobs.mutable_gpu_data();
		}else{
			out_data = top[i]->mutable_gpu_data();
		}
		const int num = bottom[i]->num();
		const int channels = bottom[i]->channels();
		const int height = bottom[i]->height();
		const int width = bottom[i]->width();
		const int top_height = top[i]->height();
		const int top_width = top[i]->width();
		// Deside the start and end index of the source according to the
		// crop info

		// The num_kernels is n*c*h
		const int  num_kernels = num * channels * height;

		bool alter_color = false;
		if (color_offset_ == 0 || color_list_[i]==false){
			alter_color = false;
		}else{
			alter_color = true;
		}

		bool is_norm_map = norm_list_[i];
		bool is_superpixel_map = superpixel_list_[i];
		bool is_pairwise_map = pairwise_list_[i];
		bool is_depth_map = depth_list_[i];
		bool is_grad_map = grad_list_[i];

		forward_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
			CAFFE_CUDA_NUM_THREADS>>>(
					num_kernels,
				   	bottom_data,
					h_offset_data,
					w_offset_data,
					color_data,
					mirror_data,
					num, 
					channels, 
					height, 
					width, 
					top_height,
					top_width,
					crop_height_,
					crop_width_,
					mirror_,
					alter_color,
					is_norm_map,
					is_superpixel_map,
					is_pairwise_map,
					is_grad_map,
					out_data);

		CUDA_POST_KERNEL_CHECK;
		// If need to crop and resize back
		if (crop_rate_ != 1){
			Dtype* top_data = top[i]->mutable_gpu_data();
			const Dtype* h_size_rate_data = batch_rand_crop_rate_.gpu_data();
			const Dtype* w_size_rate_data = batch_rand_crop_rate_.gpu_data();
			const Dtype* h_offset_rate_data = batch_rand_crop_h_offset_.gpu_data();
			const Dtype* w_offset_rate_data = batch_rand_crop_w_offset_.gpu_data();
			const Dtype* batch_rand_crop_rate_data = batch_rand_crop_rate_.cpu_data();

			cuCropAndResizeBack(out_data, top_data, h_size_rate_data, w_size_rate_data,
					h_offset_rate_data, w_offset_rate_data, num, channels, height, width);
			// If the current blob is a depth map, the depth value will be rescaled
			// according to the crop rate
			if(is_depth_map){
				for (int n = 0; n < top[i]->num(); n++){
					caffe_gpu_scal(top[i]->count(1), batch_rand_crop_rate_data[n], top_data + top[i]->offset(n));
				}
			}
		}

	} // i
}

template void SyncTransFastLayer<float>::Forward_gpu(const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top);
template void SyncTransFastLayer<double>::Forward_gpu(const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top);
//INSTANTIATE_LAYER_GPU_FUNCS(SyncTransFastLayer);

}  // namespace caffe
