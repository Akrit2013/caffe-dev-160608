// This new layer is used in fcn model, it will ignore the 'bad pixel'
// in the loss calculation.
// Define the BAD Pixel:
// If the pixel is smaller than 0 or close to 0, it will be considered as 'bad pixel'

#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/berhu_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void BerhuLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
  // Reshape the bad_pixel_
  bad_pixel_.Reshape(bottom[0]->num(), 1, bottom[0]->height(), 1); 
  // Load the parameters
  BerhuParameter berhu_param = this->layer_param_.berhu_param();
  // Load the C rate, default is 0.2
  if(berhu_param.has_c_rate()){
	c_rate_ = berhu_param.c_rate();
  }else{
	c_rate_ = 0.2;
  }
  if(berhu_param.has_h_rate()){
	h_rate_ = berhu_param.c_rate();
	has_h_rate_ = true;
  }else{
	h_rate_ = 0;
	has_h_rate_ = false;
  }
  if(berhu_param.has_max_label()){
	  has_max_label_ = true;
	  max_label_ = berhu_param.max_label();
  }else{
	  has_max_label_ = false;
	  max_label_ = 0;
  }
  if(berhu_param.has_min_label()){
	  has_min_label_ = true;
	  min_label_ = berhu_param.min_label();
  }else{
	  has_min_label_ = false;
	  min_label_ = 0;
  }
  if(berhu_param.has_invalid_label()){
	  has_invalid_label_ = true;
	  invalid_label_ = berhu_param.invalid_label();
  }else{
	  has_invalid_label_ = false;
	  invalid_label_ = 0;
  }
  if(berhu_param.has_normalize()){
	  normalize_ = berhu_param.normalize();
  }else{
	  normalize_ = false;
  }
  if(berhu_param.has_c_rate_mode()){
	  switch (berhu_param.c_rate_mode()){
		  case BerhuParameter_CRateMode_MAX:
			  c_rate_mode_ = MAX;
			  break;
		  case BerhuParameter_CRateMode_AVE:
			  c_rate_mode_ = AVE;
			  break;
		  default:
			  LOG(FATAL) << "Unsupport c_rate_mode";
			  break;
	  }
  }else{
	  c_rate_mode_ = MAX;
  }
}

template <typename Dtype>
void BerhuLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  Dtype max_diff = 0;

  switch(c_rate_mode_){
	  case MAX:
		  // Get the abs max diff to determine the C
		  max_diff = caffe_amax(count, diff_.cpu_data(), 1);
		  // Calc the Threshold C
		  break;
	  case AVE:
		  // Calc the mean of the abs diff
		  max_diff = caffe_cpu_asum(count, diff_.cpu_data()) / count;
		  break;
	  default:
		  LOG(FATAL) << "False c_rate_mode";
		  break;
  }
  Dtype C = fabs(max_diff * c_rate_);
  Dtype H = fabs(max_diff * h_rate_);
  // For debug
  // LOG(INFO)<<"Max_diff:"<<max_diff<<" C:"<<C;
  // Iter the diff map
  Dtype* data_diff = diff_.mutable_cpu_data();
  // const Dtype* data_pred = bottom[0]->cpu_data();
  // Set the diff to zero if label pixel is zero (all channel is zero)
  int bad_pixel_count = 0;
  const Dtype* data_label = bottom[1]->cpu_data();
  for(int n = 0; n < diff_.num(); ++n){
	 for(int h = 0; h < diff_.height(); ++h){
		  for(int w = 0; w < diff_.width(); ++w){
			  int err_counter = 0;
			  for(int c = 0; c < diff_.channels(); ++c){
				  int index = ((n*diff_.channels()+c)*diff_.height()+h)*diff_.width()+w;
				  Dtype dataval = data_label[index];
				  if (has_max_label_ && dataval > max_label_){
					  err_counter++;
				  }else if(has_min_label_ && dataval < min_label_){
					  err_counter++;
				  }else if(has_invalid_label_ && fabs(dataval - invalid_label_) < 0.0001){
					  err_counter++;
				  }
				  // Set the diff to L1 or L2 according to the C
				  Dtype diff_val = data_diff[index];
				  if (fabs(diff_val) <= C){
					  // L1
					  if (diff_val > 0){
						  data_diff[index] = C;
					  }else if(diff_val < 0){
						  data_diff[index] = -C;
					  }
				  }else if(has_h_rate_ && fabs(diff_val) > H){
					  if (diff_val > 0){
						  data_diff[index] = H;
					  }else{
						  data_diff[index] = -H;
					  }
				  }
			  }
			  if(err_counter == diff_.channels()){
				  // This pixel is not ok
				  bad_pixel_count += diff_.channels();
				  for(int c = 0; c < diff_.channels(); ++c){
					  int index = ((n*diff_.channels()+c)*diff_.height()+h)*diff_.width()+w;
					  data_diff[index] = 0;
				  }
			  }
		  }
	 }
  }

  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / Dtype(2) / (count - bad_pixel_count);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void BerhuLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(BerhuLossLayer);
#endif

INSTANTIATE_CLASS(BerhuLossLayer);
REGISTER_LAYER_CLASS(BerhuLoss);

}  // namespace caffe
