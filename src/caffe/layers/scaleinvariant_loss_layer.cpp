#include <vector>
#include <cfloat>
#include "caffe/layers/scaleinvariant_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ScaleInvariantLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  CHECK_EQ(bottom[0]->channels(), 1)
	  << "Inputs channels must is 1";
  diff_.ReshapeLike(*bottom[0]);
  // Reshape the bad_pixel_
  bad_pixel_.Reshape(bottom[0]->num(), 1, bottom[0]->height(), 1);
  // Load the parameter
  ScaleInvariantParameter scale_invariant_param = this->layer_param_.scale_invariant_param();
  // Load the delta
  if(scale_invariant_param.has_delta()){
	  delta_ = scale_invariant_param.delta();
  }
  // Load the min and max value
  if(scale_invariant_param.has_min_label()){
	  min_val_ = scale_invariant_param.min_label();
  }else{
	  min_val_ = Dtype(FLT_MIN);
  }
  if(scale_invariant_param.has_max_label()){
	  max_val_ = scale_invariant_param.max_label();
  }else{
	  max_val_ = Dtype(FLT_MAX);
  }
  if(scale_invariant_param.has_max_label() || scale_invariant_param.has_min_label()){
	  is_use_bad_pixel_ = true;
  }else{
	  is_use_bad_pixel_ = false;
  }
  if(scale_invariant_param.has_adjust_pixel_num()){
	  is_adjust_pixel_num_ = scale_invariant_param.adjust_pixel_num();
  }else{
	  is_adjust_pixel_num_ = true;
  }
  // Resize the vecSum_
  vecSum_.Reshape(bottom[0]->num(), 1, 1, 1);
  vecValidPixelNum_.Reshape(bottom[0]->num(), 1, 1, 1);
  valid_pixel_num_ = 0;
}

template <typename Dtype>
void ScaleInvariantLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());

  valid_pixel_num_ = 0;
  Dtype* vecValidPixelNum_data = vecValidPixelNum_.mutable_cpu_data();
  if (is_use_bad_pixel_ == true){
	  const Dtype* data_label = bottom[1]->cpu_data();
	  Dtype* data_diff = diff_.mutable_cpu_data();
	  for(int n = 0; n < diff_.num(); n++){
		  // Reset the valid pixel number of each n
		  vecValidPixelNum_data[n] = 0;
		  // The channels must be 1 here
		  for(int c = 0; c < diff_.channels(); c++){
			  for(int h = 0; h < diff_.height(); h++){
				  const Dtype* data_label_w = data_label + bottom[1]->offset(n, c, h);
				  Dtype* data_diff_w = data_diff + diff_.offset(n, c, h);
				  for(int w = 0; w < diff_.width(); w++){
					  Dtype val = data_label_w[w];
					  if (val >= min_val_ && val <= max_val_){
						  valid_pixel_num_++;
						  vecValidPixelNum_data[n]++;
					  }else{
						  // Set the diff to 0
						  data_diff_w[w] = Dtype(0);
					  }
				  }
			  }
		  }
	  }
  }else{
	  valid_pixel_num_ = bottom[0]->count();
	  for(int n = 0; n < bottom[0]->num(); n++){
		  // Set the valid pixel
		  vecValidPixelNum_data[n] = bottom[0]->count(1);
	  }
  }
  // If not adjust the pixel number, reset the pixel number
  if (is_adjust_pixel_num_ == false){
	  valid_pixel_num_ = bottom[0]->count();
	  for(int n = 0; n < bottom[0]->num(); n++){
		  // Set the valid pixel
		  vecValidPixelNum_data[n] = bottom[0]->count(1);
	  }
  }

  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  // The first term of the loss, basically is the L2 loss
  Dtype loss = dot / valid_pixel_num_ / Dtype(2);

  // Calc the second term of the loss
  // Calc the sum of the diff
  // The tmp vector for the sum of the each sample for the minibatch
  // The pixels of the image
  Dtype pixel_num = Dtype(bottom[0]->count(1));
  Dtype* vecSum_data = vecSum_.mutable_cpu_data();
  for (int n = 0; n < bottom[0]->num(); n++){
	  const Dtype* pdata = diff_.cpu_data() + bottom[0]->offset(n);
	  Dtype valid_num = vecValidPixelNum_data[n];
	  vecSum_data[n] = caffe_cpu_sum(pixel_num, pdata);
	  loss -= vecSum_data[n] * vecSum_data[n] / valid_num / valid_num / bottom[0]->num() * delta_ / Dtype(2);
  }

  top[0]->mutable_cpu_data()[0] = loss;
  DLOG(INFO)<<"pred:"<<bottom[0]->cpu_data()[0];
  DLOG(INFO)<<"label:"<<bottom[1]->cpu_data()[0];
  DLOG(INFO)<<"loss:"<<loss;
}

template <typename Dtype>
void ScaleInvariantLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* vecSum_data = vecSum_.cpu_data();
  const Dtype* vecValidPixelNum_data = vecValidPixelNum_.cpu_data();
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
	  // Calc the L2 term of the diff
      const Dtype sign = (i == 0) ? 1 : -1;
//      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->count();
//      caffe_cpu_axpby(
//          bottom[i]->count(),              // count
//          alpha,                              // alpha
//          diff_.cpu_data(),                   // a
//          Dtype(0),                           // beta
//          bottom[i]->mutable_cpu_diff());  // b
	  // Calc the second term of the diff
	  const Dtype* diff_data = diff_.cpu_data();
	  Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
	  for(int n = 0; n < bottom[0]->num(); n++){
		  Dtype valid_num = vecValidPixelNum_data[n];
		  const Dtype w1 = Dtype(1) / valid_pixel_num_;
		  const Dtype w2 = vecSum_data[n] * delta_ / bottom[0]->num() / valid_num / valid_num;
		  // sign * w1 * diff - sign * w2
		  // There must be a faster way to perform this,  YanHan
		  int offset = diff_.offset(n);
		  for (int j = 0; j < diff_.count(1); j++){
			  bottom_diff[offset + j] = sign * top[0]->cpu_diff()[0] * (w1 * diff_data[offset + j] - w2);
		  }
	  }
		  
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ScaleInvariantLossLayer);
#endif

INSTANTIATE_CLASS(ScaleInvariantLossLayer);
REGISTER_LAYER_CLASS(ScaleInvariantLoss);

}  // namespace caffe
