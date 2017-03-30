#include <cfloat>
#include <vector>

#include "caffe/layers/fusion_l2max_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FusionL2MaxLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void FusionL2MaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK(bottom[i]->shape() == bottom[0]->shape());
  }
  top[0]->ReshapeLike(*bottom[0]);
  max_idx_.Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void FusionL2MaxLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int* mask_data = max_idx_.mutable_cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int bottom_size = bottom.size();
  caffe_set(max_idx_.count(), 0, mask_data);
  caffe_set(top[0]->count(), Dtype(0), top_data);
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  const int plan_size = height * width;

  for (int i = 0; i < bottom_size; i++){
	  const Dtype* bottom_data = bottom[i]->cpu_data();
	  for (int n = 0; n < num; n++){
		  for (int h = 0; h < height; h++){
			  for (int w = 0; w < width; w++){
				  // The length of the current pixel'
				  Dtype len_bottom = 0;
				  Dtype len_top = 0;
				  int idx_top = top[0]->offset(n, 0, h, w);
				  int idx_mask = max_idx_.offset(n, 0, h, w);
				  for (int c = 0; c < channels; c++){
					  Dtype val_bottom = bottom_data[idx_top + c * plan_size];
					  Dtype val_top = top_data[idx_top + c * plan_size];
					  len_top += val_top * val_top;
					  len_bottom += val_bottom * val_bottom;
				  }
				  if (len_bottom > len_top){
					  // Replace the top value with the bottom value
					  for (int c = 0; c < channels; c++){
						  top_data[idx_top + c * plan_size] = bottom_data[idx_top + c*plan_size];
					  }
					  mask_data[idx_mask] = i;
				  }
			  }
		  }
	  }
  }
					  
}

template <typename Dtype>
void FusionL2MaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const int* mask_data = max_idx_.cpu_data();

  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
	  // Clear the previous diff
	  caffe_set(bottom[i]->count(), Dtype(0), bottom_diff);
	  int plan_size = bottom[i]->height() * bottom[i]->width();

	  for (int n = 0; n < bottom[i]->num(); n++){
		  for (int h = 0; h < bottom[i]->height(); h++){
			  for (int w = 0; w < bottom[i]->width(); w++){
				  int idx_mask = max_idx_.offset(n, 0, h, w);
				  if (mask_data[idx_mask] == i){
					  int idx_top = top[0]->offset(n, 0, h, w);
					  for (int c = 0; c < bottom[i]->channels(); c++){
						  bottom_diff[idx_top + c*plan_size] = top_diff[idx_top+c*plan_size];
					  }
				  }
			  }
		  }
	  }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(FusionL2MaxLayer);
#endif

INSTANTIATE_CLASS(FusionL2MaxLayer);
REGISTER_LAYER_CLASS(FusionL2Max);

}  // namespace caffe
