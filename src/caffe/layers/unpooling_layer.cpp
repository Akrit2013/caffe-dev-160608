/*************************************************************************
	> File Name: unpooling_layer.cpp
	> Author:
	> Mail:
	> Created Time: 2015年11月16日 星期一 15时42分28秒
NOTE:
1. Currently, Only support MAX and AVE unpooling method
2. The strider and the kernal size must be the same (if both are set)
3. The pad is no longer use 0, but use the pattern determined by the unpooling
   method.
NOTE:
1. In the MAX unpooling, we only use the left up pixel as the original pixel
   but put other pixels to 0.
2. In the AVE unpooling, we simple copy the pixel value to the unpooled positions
NOTE:
For Backward computation
1. In the MAX unpooling, we only use the upper-left corner position of each
   pooling kernel's diff.
2. In the AVE unpooling, we use the average diff in a kernel as the BP diff.
 ************************************************************************/

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/unpooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void UnpoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  UnpoolingParameter unpool_param = this->layer_param_.unpooling_param();
  // Currently, only support 1 bottom and 1 top
  CHECK(bottom.size()==1) << "The UnpoolingLayer only can have 1 bottom blob";
  CHECK(top.size()==1) << "The UnpoolingLayer only can have 1 top blob";

  // Check the parameters
  CHECK(!(unpool_param.has_pad() && (unpool_param.has_pad_u() || unpool_param.has_pad_d() || unpool_param.has_pad_l() || unpool_param.has_pad_r())))
	  << "pad, and pad_u, pad_d, pad_r, pad_l can not have both";

  CHECK(!unpool_param.has_kernel_size() !=
		  !(unpool_param.has_kernel_h() && unpool_param.has_kernel_w()))
	  << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK(unpool_param.has_kernel_size() ||
		  (unpool_param.has_kernel_h() && unpool_param.has_kernel_w()))
	  << "For non-square filters both kernel_h and kernel_w are required.";
  CHECK((!unpool_param.has_stride() && unpool_param.has_stride_h()
      && unpool_param.has_stride_w())
      || (!unpool_param.has_stride_h() && !unpool_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  if (unpool_param.has_kernel_size()) {
	  kernel_h_ = kernel_w_ = unpool_param.kernel_size();
  } else {
	  kernel_h_ = unpool_param.kernel_h();
	  kernel_w_ = unpool_param.kernel_w();
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (unpool_param.has_pad()) {
	  pad_u_ = pad_d_ = pad_l_ = pad_r_ = unpool_param.pad();
  } else {
    pad_u_ = unpool_param.pad_u();
    pad_d_ = unpool_param.pad_d();
    pad_l_ = unpool_param.pad_l();
    pad_r_ = unpool_param.pad_r();
  }
  if (!unpool_param.has_stride_h()) {
    stride_h_ = stride_w_ = unpool_param.stride();
  } else {
    stride_h_ = unpool_param.stride_h();
    stride_w_ = unpool_param.stride_w();
  }

  // kernel and stride must be the same
  CHECK(kernel_h_ == stride_h_) << "Kernel and Strider must be same";
  CHECK(kernel_w_ == stride_w_) << "Kernel and Strider must be same";
}


template <typename Dtype>
void UnpoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  unpooled_height_ = static_cast<int>((height_ - 1) * stride_h_ + kernel_h_ + pad_u_ + pad_d_);
  unpooled_width_ = static_cast<int>((width_ - 1) * stride_w_ + kernel_w_ + pad_l_ + pad_r_);
  top[0]->Reshape(bottom[0]->num(), channels_, unpooled_height_,
      unpooled_width_);
}



template <typename Dtype>
void UnpoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype *bottom_data = bottom[0]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();
  // Init the top blob to be all zero since only special places
  // wouldn't be zero then.
  caffe_set(top[0]->count(), Dtype(0), top_data);

  switch (this->layer_param_.unpooling_param().pool()) {
  case UnpoolingParameter_UnpoolMethod_MAX:
      for (int n = 0; n < bottom[0]->num(); ++n) {
          for (int c = 0; c < channels_; ++c) {
              for (int ph = 0; ph < height_; ++ph) {
                  for (int pw = 0; pw < width_; ++pw) {
					  // Calc the index in the unpooled map
				      const int index = ph * width_ + pw;
					  const int uph = pad_u_ + (ph*stride_h_);
					  const int upw = pad_l_ + (pw*stride_w_);
					  // Check if out of border
					  if(uph<0 || uph>=unpooled_height_ || upw<0 || upw>=unpooled_width_){
						  continue;
					  }
					  const int top_index = uph*unpooled_width_+upw;
                      top_data[top_index] = bottom_data[index];
                  }
              }
              //switch to next channel
              top_data += top[0]->offset(0, 1);
              bottom_data += bottom[0]->offset(0, 1);
          }
      }
      break;
  case UnpoolingParameter_UnpoolMethod_AVE:
	  for (int n = 0; n < bottom[0]->num(); ++n) {
          for (int c = 0; c < channels_; ++c) {
              for (int ph = 0; ph < height_; ++ph) {
                  for (int pw = 0; pw < width_; ++pw) {
					  // Calc the index in the unpooled map
				      const int index = ph * width_ + pw;
					  // Loop the kernel patch
					  const int uph_offset = pad_u_ + (ph*stride_h_);
					  const int upw_offset = pad_l_ + (pw*stride_w_);
					  const Dtype val = bottom_data[index];
					  // Cover the edge
					  int upw_start = upw_offset;
					  int upw_end = upw_offset + kernel_w_;
					  if (pw==0){
						  upw_start = 0;
					  }
					  if (pw==width_-1){
						  upw_end = unpooled_width_;
					  }
					  int uph_start = uph_offset;
					  int uph_end = uph_offset + kernel_h_;
					  if (ph==0){
						  uph_start = 0;
					  }
					  if (ph==height_-1){
						  uph_end = unpooled_height_;
					  }
					  for (int upw = upw_start; upw < upw_end; upw++){
						  if(upw<0 || upw>=unpooled_width_) continue;
						  for(int uph=uph_start; uph < uph_end; uph++){
							  if(uph<0 || uph>=unpooled_height_) continue;
							  // Set the data
							  const int top_index = uph*unpooled_width_+upw;
							  top_data[top_index] = val;
						  }
					  }

                  }
              }
              //switch to next channel
              top_data += top[0]->offset(0, 1);
              bottom_data += bottom[0]->offset(0, 1);
          }
      }
      break;
  }

}


template <typename Dtype>
void  UnpoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (!propagate_down[0]) {
		return;
	}
	const Dtype* top_diff = top[0]->cpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	// Different pooling methods. We explicitly do the switch outside the for
	// loop to save time, although this results in more codes.
	caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
	switch (this->layer_param_.unpooling_param().pool()) {
		case UnpoolingParameter_UnpoolMethod_MAX:
			for (int n = 0; n < bottom[0]->num(); ++n) {
				for (int c = 0; c < channels_; ++c) {
					for (int ph = 0; ph < height_; ++ph) {
						for (int pw = 0; pw < width_; ++pw) {
							// Calc the index in the unpooled map
							const int index = ph * width_ + pw;
							const int uph = pad_u_ + (ph*stride_h_);
							const int upw = pad_l_ + (pw*stride_w_);
							// Check if out of border
							if(uph<0 || uph>=unpooled_height_ || upw<0 || upw>=unpooled_width_){
								continue;
							}
							const int top_index = uph*unpooled_width_+upw;
							// LOG(INFO)<<"ph: "<<ph<<"uph:"<<uph<<"pw:"<<pw<<"upw:"<<upw<<"VAL:"<<top_diff[top_index];
							bottom_diff[index] = top_diff[top_index];
						}
					}
					//switch to next channel
					top_diff += top[0]->offset(0, 1);
					bottom_diff += bottom[0]->offset(0, 1);
				}
			}
			break;
		case UnpoolingParameter_UnpoolMethod_AVE:
			for (int n = 0; n < bottom[0]->num(); ++n) {
				for (int c = 0; c < channels_; ++c) {
					for (int ph = 0; ph < height_; ++ph) {
						for (int pw = 0; pw < width_; ++pw) {
							// Calc the index in the unpooled map
							const int index = ph * width_ + pw;
							// Loop the kernel patch
							const int upw_offset = pad_l_ + (pw*stride_w_);
							const int uph_offset = pad_u_ + (ph*stride_h_);
							// Cover the edge
							int upw_start = upw_offset;
							int upw_end = upw_offset + kernel_w_;
							if (pw==0){
								upw_start = 0;
							}
							if (pw==width_-1){
								upw_end = unpooled_width_;
							}
							int uph_start = uph_offset;
							int uph_end = uph_offset + kernel_h_;
							if (ph==0){
								uph_start = 0;
							}
							if (ph==height_-1){
								uph_end = unpooled_height_;
							}
							Dtype val_accu = 0;
							int val_counter = 0;
							for (int upw = upw_start; upw < upw_end; upw++){
								if(upw<0 || upw>=unpooled_width_) continue;
								for(int uph=uph_start; uph < uph_end; uph++){
									if(uph<0 || uph>=unpooled_height_) continue;
									// Set the data
									const int top_index = uph*unpooled_width_+upw;
									val_accu += top_diff[top_index];
									val_counter++;
								}
							}
							Dtype val = val_accu / val_counter;
							bottom_diff[index] = val;
						}
					}
					//switch to next channel
					top_diff += top[0]->offset(0, 1);
					bottom_diff += bottom[0]->offset(0, 1);
				}
			}
			break;
		default:
			LOG(FATAL) << "Unknown pooling method.";
	}
}

//We need only CPU version.
// STUB_GPU(UnPoolingLayer);

INSTANTIATE_CLASS(UnpoolingLayer);
REGISTER_LAYER_CLASS(Unpooling);
} //namespace caffe
