#include <vector>
#include <cfloat>

#include "caffe/layers/superpixel_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SuperpixelPoolingLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	// This layer has no params for now
	// Check the blobs
	CHECK_EQ(bottom[0]->num(), bottom[1]->num());
	// Currently, the channels of the bottom[1] must be 1
	CHECK_EQ(bottom[1]->channels(), 1);

	// Parse the params
	SuperpixelPoolingParameter superpixel_param = this->layer_param_.superpixel_pooling_param();
	// NOTE: The num_output_ must be equal with the number of superpixels
	if (superpixel_param.has_num_output()){
		num_output_ = superpixel_param.num_output();
	}else{
		LOG(FATAL) << "The num_output must be set equal to the number of the superpixels";
	}

	if (superpixel_param.has_pool()){
		switch (superpixel_param.pool()){
			case SuperpixelPoolingParameter_PoolMethod_MAX:
				pool_method_ = MAX;
				break;
			case SuperpixelPoolingParameter_PoolMethod_AVE:
				pool_method_ = AVE;
				break;
			case SuperpixelPoolingParameter_PoolMethod_STOCHASTIC:
				NOT_IMPLEMENTED;
				pool_method_ = STOCHASTIC;
				break;
			default:
				NOT_IMPLEMENTED;
				break;
		}
	}else{
		pool_method_ = AVE;
	}
}

template <typename Dtype>
void SuperpixelPoolingLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	// Reshape the top
	top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), num_output_, 1);
	// Constuct the blob to store the relationship between the bottom[0] and the
	// superpixel label
	mask_.ReshapeLike(*bottom[0]);
	accum_.ReshapeLike(*top[0]);
	
	// Init the params
	num_ = bottom[0]->num();
	sp_height_ = bottom[1]->height();
	sp_width_ = bottom[1]->width();
	in_channels_ = bottom[0]->channels();
	in_height_ = bottom[0]->height();
	in_width_ = bottom[0]->width();
	h_rate_ = (Dtype)sp_height_ / in_height_;
	w_rate_ = (Dtype)sp_width_ / in_width_;
}

template <typename Dtype>
void SuperpixelPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const Dtype* sp_data = bottom[1]->cpu_data();
	const Dtype* in_data = bottom[0]->cpu_data();
	Dtype* out_data = top[0]->mutable_cpu_data();
	Dtype* mask_data = mask_.mutable_cpu_data();
	Dtype* accum_data = accum_.mutable_cpu_data();

	// Clear the accumulate memory to zero
	caffe_set(accum_.count(), Dtype(0), accum_data);
	caffe_set(top[0]->count(), Dtype(0), out_data);

	// Iter the in_data to find the
	switch (pool_method_){
		case AVE:
			caffe_set(top[0]->count(), Dtype(0), out_data);
			for (int n = 0; n < num_; n++){
				const Dtype* sp_data_n = sp_data + bottom[1]->offset(n);
				for (int c = 0; c < in_channels_; c++){
					const Dtype* in_data_nc = in_data + bottom[0]->offset(n, c);
					Dtype* out_data_nc = out_data + top[0]->offset(n, c);
					Dtype* accum_data_nc = accum_data + accum_.offset(n, c);
					Dtype* mask_data_nc = mask_data + mask_.offset(n, c);

					for (int h = 0; h < in_height_; h++){
						for (int w = 0; w < in_width_; w++){
							// Calc the corresponding position
							const int h_c = round(h_rate_*(h+0.5));
							const int w_c = round(w_rate_*(w+0.5));
							const int sp_index = h_c*sp_width_+w_c;
							const int in_index = h*in_width_+w;
							const int out_index = (int)sp_data_n[sp_index];
							out_data_nc[out_index] += in_data_nc[in_index];
							accum_data_nc[out_index] += 1;
							mask_data_nc[in_index] = out_index;

						}
					}
				}
			}

			// Check which superpixel have not got the valid value, can put the
			// neareat bottom value to it
			// TODO: Since the special info has been lost in the top blob
			// when the invaild superpixel is on edge, that will cause a problem
			for (int n = 0; n < num_; n++){
				for (int c = 0; c < in_channels_; c++){
					for (int hw = 1; hw < num_output_-1; hw++){
						int idx = accum_.offset(n, c, hw);
						if(accum_data[idx] == 0){
							out_data[idx] = out_data[idx+1] + out_data[idx-1];
							accum_data[idx] = accum_data[idx+1] + accum_data[idx-1];
						}
					}
				}
			}
			// Average
			// NOTE: To avoid the NaN which caused by the 0 in accum_data, all 0 must be set to 1
			for (int j = 0; j < accum_.count(); j++){
				if(accum_data[j] == 0){
					accum_data[j] = 1;
				}
			}
			// TODO: This is a in-place operation, might cause a problem
			caffe_div(top[0]->count(), out_data, accum_data, out_data);

			break;
		case MAX:
			caffe_set(top[0]->count(), Dtype(FLT_MIN), out_data);
			for (int n = 0; n < num_; n++){
				const Dtype* sp_data_n = sp_data + bottom[1]->offset(n);
				for (int c = 0; c < in_channels_; c++){
					const Dtype* in_data_nc = in_data + bottom[0]->offset(n, c);
					Dtype* out_data_nc = out_data + top[0]->offset(n, c);
					Dtype* accum_data_nc = accum_data + accum_.offset(n, c);

					for (int h = 0; h < in_height_; h++){
						for (int w = 0; w < in_width_; w++){
							// Calc the corresponding position
							const int h_c = round(h_rate_*(h+0.5));
							const int w_c = round(w_rate_*(w+0.5));
							const int sp_index = h_c*sp_width_+w_c;
							const int in_index = h*in_width_+w;
							const int out_index = (int)sp_data_n[sp_index];
							if (out_data_nc[out_index] < in_data_nc[in_index]){
								out_data_nc[out_index] = in_data_nc[in_index];
								accum_data_nc[out_index] = in_index;
							}
						}
					}
				}
			}
			break;
		case STOCHASTIC:
			NOT_IMPLEMENTED;
			break;
		default:
			NOT_IMPLEMENTED;
			break;
	}
}

template<typename Dtype>
void SuperpixelPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
	if (!propagate_down[0]){return;}

	const Dtype* top_diff = top[0]->cpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	const Dtype* mask_data = mask_.cpu_data();
	const Dtype* accum_data = accum_.cpu_data();

	// Clear the diff
	caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

	// Iter the bottom
	switch (pool_method_){
		case AVE:
			for(int n = 0; n < num_; n++){
				for (int c = 0; c < in_channels_; c++){
					const Dtype* top_diff_nc = top_diff + top[0]->offset(n, c);
					const Dtype* mask_data_nc = mask_data + mask_.offset(n, c);
					Dtype* bottom_diff_nc = bottom_diff + bottom[0]->offset(n, c);

					for (int h = 0; h < in_height_; h++){
						for (int w = 0; w < in_width_; w++){
							const int bottom_index = h * in_width_ + w;
							const int top_index = static_cast<int>(mask_data_nc[bottom_index]);
							bottom_diff_nc[bottom_index] = top_diff_nc[top_index];
						}
					}
				}
			}
			break;
		case MAX:
			for(int n = 0; n < num_; n++){
				for (int c = 0; c < in_channels_; c++){
					const Dtype* top_diff_nc = top_diff + top[0]->offset(n, c);
					const Dtype* accum_data_nc = accum_data + accum_.offset(n, c);
					Dtype* bottom_diff_nc = bottom_diff + bottom[0]->offset(n, c);

					for (int i = 0; i < num_output_; i++){
						const int bottom_index = static_cast<int>(accum_data_nc[i]);
						bottom_diff_nc[bottom_index] = top_diff_nc[i];
					}
				}
			}
			break;
		case STOCHASTIC:
			NOT_IMPLEMENTED;
			break;
		default:
			NOT_IMPLEMENTED;
			break;
	}
}

INSTANTIATE_CLASS(SuperpixelPoolingLayer);
REGISTER_LAYER_CLASS(SuperpixelPooling);

}  // namespace caffe
