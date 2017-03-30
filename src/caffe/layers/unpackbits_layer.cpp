#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/unpackbits_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void UnpackBitsLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  UnpackBitsParameter unpackbits_param = this->layer_param_.unpackbits_param();
  // Check the param
  if (unpackbits_param.has_channels()){
	  channels_ = unpackbits_param.channels();
  }else{
	  channels_ = 16;
  }
  if (unpackbits_param.has_compact()){
	  compact_ = unpackbits_param.compact();
  }
  CHECK_EQ(bottom.size(), 1) << "Only can take 1 bottom layer";
  CHECK_EQ(top.size(), 1) << "Only can take 1 top layer";
  // TODO: Currently, only support the input blobs contains 1 channel
  CHECK_EQ(bottom[0]->channels(), 1) << "Currently, the bottom layer only can have 1 channel";

  if (compact_){
	  CHECK_LE(channels_, 16) << "Max channels is 16";
  }else{
	  CHECK_LE(channels_, 8) << "Max channels is 8";
  }
}

template <typename Dtype>
void UnpackBitsLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(), channels_, bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void UnpackBitsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  int channel_offset = bottom[0]->height() * bottom[0]->width();
  // Set the top to zero
  caffe_set(top[0]->count(), Dtype(0), top_data);

  // Loop the bottom
  if (compact_) {
	  // If use compact storage
	  for (int n = 0; n < bottom[0]->num(); ++n) {
		  for (int c = 0; c < bottom[0]->channels(); ++c) {
			  // Section 0, h > w
			  int s_iter = (channels_+1)/2;
			  for (int h = 1; h < bottom[0]->height(); ++h) {
				  for (int w = 0; w < h; ++w) {
					  int index = ((n*bottom[0]->channels()+c)*bottom[0]->height()+h)*bottom[0]->width()+w;
					  int top_index = top[0]->offset(n, 0, h, w);
					  int top_index2 = top[0]->offset(n, 0, w, h);
					  uint8_t val = static_cast<uint8_t>(bottom_data[index]);

					  for (int idx = 0; idx < s_iter; idx++){
						  int target_val = (val & (1u << idx)) >> idx;
						  // Since the target feature map is synmmtic
						  // TODO: If the channels is not 1, there will be a problem
						  top_data[top_index+channel_offset*idx*2] = static_cast<Dtype>(target_val);
						  top_data[top_index2+channel_offset*idx*2] = static_cast<Dtype>(target_val);
					  }
				  }
			  }
			  // Section 1, h < w
			  s_iter = channels_/2;
			  for (int w = 1; w < bottom[0]->width(); ++w) {
				  for (int h = 0; h < w; ++h) {
					  int index = ((n*bottom[0]->channels()+c)*bottom[0]->height()+h)*bottom[0]->width()+w;
					  int top_index = top[0]->offset(n, 0, h, w);
					  int top_index2 = top[0]->offset(n, 0, w, h);
					  uint8_t val = static_cast<uint8_t>(bottom_data[index]);

					  for (int idx = 0; idx < s_iter; idx++){
						  int target_val = (val & (1u << idx)) >> idx;
						  top_data[top_index+channel_offset*(idx*2+1)] = static_cast<Dtype>(target_val);
						  top_data[top_index2+channel_offset*(idx*2+1)] = static_cast<Dtype>(target_val);
					  }
				  }
			  }
		  }
	  }
  }else{
	  // Not compact
	  for (int n = 0; n < bottom[0]->num(); ++n) {
		  for (int c = 0; c < bottom[0]->channels(); ++c) {
			  for (int h = 1; h < bottom[0]->height(); ++h) {
				  for (int w = 0; w < h; ++w) {
					  int index = ((n*bottom[0]->channels()+c)*bottom[0]->height()+h)*bottom[0]->width()+w;
					  int top_index = top[0]->offset(n, 0, h, w);
					  int top_index2 = top[0]->offset(n, 0, w, h);
					  uint8_t val = static_cast<uint8_t>(bottom_data[index]);

					  for (int idx = 0; idx < channels_; idx++){
						  int target_val = (val & (1u << idx)) >> idx;
						  // Since the target feature map is synmmtic
						  top_data[top_index+channel_offset*idx] = static_cast<Dtype>(target_val);
						  top_data[top_index2+channel_offset*idx] = static_cast<Dtype>(target_val);
					  }
				  }
			  }
		  }
	  }
  }
}

#ifdef CPU_ONLY
STUB_GPU(UnpackBitsLayer);
#endif

INSTANTIATE_CLASS(UnpackBitsLayer);
REGISTER_LAYER_CLASS(UnpackBits);

}  // namespace caffe
