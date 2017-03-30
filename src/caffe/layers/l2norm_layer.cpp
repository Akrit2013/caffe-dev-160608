#include <algorithm>
#include <vector>

#include "caffe/layers/l2norm_layer.hpp"

namespace caffe {

template <typename Dtype>
void L2NormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(bottom.size(), 1);
	CHECK_GT(bottom[0]->channels(), 1) << "Must have at least 2 channels when perform normalization according channels";
	// Check the parameter
	L2NormParameter l2norm_param = this->layer_param().l2norm_param();
	if (l2norm_param.has_l2_length()){
		norm_val_ = l2norm_param.l2_length();
	}else{
		norm_val_ = 1;
	}
}

template <typename Dtype>
void L2NormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	top[0]->ReshapeLike(*bottom[0]);
	// Reshape the length_map_
	length_map_.Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
}

	
template <typename Dtype>
void L2NormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* len_data = length_map_.mutable_cpu_data();
  caffe_set(length_map_.count(), Dtype(0), len_data);
  // Calc the l2 length of each pixel
  for (int n = 0; n < bottom[0]->num(); n++){
	  for (int h = 0; h < bottom[0]->height(); h++){
		  for (int w = 0; w < bottom[0]->width(); w++){
			  int len_idx = length_map_.offset(n, 0, h, w);
			  for (int c = 0; c < bottom[0]->channels(); c++){
				  Dtype val = bottom_data[bottom[0]->offset(n,c,h,w)];
				  len_data[len_idx] += val*val/norm_val_;
			  }
			  if (len_data[len_idx] == 0){
				  len_data[len_idx] = 1;
			  }else{
				  len_data[len_idx] = sqrt(len_data[len_idx]);
			  }
		  }
	  }
  }
  // Normalize the data
  for (int n = 0; n < bottom[0]->num(); n++){
	  Dtype* len_data_n = len_data + length_map_.offset(n);
	  for (int c = 0; c < bottom[0]->channels(); c++){
		  const Dtype* bottom_data_nc = bottom_data + bottom[0]->offset(n, c);
		  Dtype* top_data_nc = top_data + top[0]->offset(n, c);
		  int count = bottom[0]->height() * bottom[0]->width();
		  caffe_div(count, bottom_data_nc, len_data_n, top_data_nc);
	  }
  }
}

template <typename Dtype>
void L2NormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
	const Dtype* len_data = length_map_.cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count(2);

	for (int n = 0; n < bottom[0]->num(); n++){
		const Dtype* len_data_n = len_data + length_map_.offset(n);
		for (int c = 0; c < bottom[0]->channels(); c++){
			const Dtype* top_diff_nc = top_diff + top[0]->offset(n,c);
			Dtype* bottom_diff_nc = bottom_diff + bottom[0]->offset(n,c);
			caffe_div(count, top_diff_nc, len_data_n, bottom_diff_nc);
		}
	}
  }
}


#ifdef CPU_ONLY
STUB_GPU(L2NormLayer);
#endif

INSTANTIATE_CLASS(L2NormLayer);
REGISTER_LAYER_CLASS(L2Norm);

}  // namespace caffe
