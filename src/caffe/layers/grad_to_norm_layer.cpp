#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/grad_to_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void GradToNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Current, it contains no param
  // Check the top and bottom
  CHECK_EQ(bottom[0]->channels(), 2) <<"The bottom should be a gradient map with 2 channels";
}

template <typename Dtype>
void GradToNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  top[0]->Reshape(bottom[0]->num(), 3, bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void GradToNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int stride = bottom[0]->count(2);

  // Iter the pixels
  for (int n = 0; n < bottom[0]->num(); n++){
	  for (int h = 0; h < bottom[0]->height(); h++){
		  for (int w = 0; w < bottom[0]->width(); w++){
			  const int bottom_idx = bottom[0]->offset(n, 0, h, w);
			  const int top_idx = top[0]->offset(n, 0, h, w);
			  const Dtype dx = bottom_data[bottom_idx];
			  const Dtype dy = bottom_data[bottom_idx + stride];

			  Dtype z = 1;
			  Dtype x = - dx * z;
			  Dtype y = - dy * z;

			  // Normalize the [x, y, z]
			  const Dtype norm = sqrt(x*x+y*y+z*z);
			  if (norm == 0){
				  x = 0;
				  y = 0;
				  z = 0;
			  }else{
				  x /= norm;
				  y /= norm;
				  z /= norm;
			  }
			  top_data[top_idx] = x;
			  top_data[top_idx + stride] = y;
			  top_data[top_idx + stride * 2] = z;
		  }
	  }
  }

}

template <typename Dtype>
void GradToNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

  for (int n = 0; n < bottom[0]->num(); n++){
	  for (int h = 0; h < bottom[0]->height(); h++){
		  for (int w = 0; w < bottom[0]->width(); w++){
			  const Dtype z = top_data[top[0]->offset(n, 2, h, w)];
			  const Dtype diff_dx = - top_diff[top[0]->offset(n, 0, h, w)] / z;
			  const Dtype diff_dy = - top_diff[top[0]->offset(n, 1, h, w)] / z;

			  bottom_diff[bottom[0]->offset(n, 0, h, w)] = diff_dx;
			  bottom_diff[bottom[0]->offset(n, 1, h, w)] = diff_dy;
		  }
	  }
  }
}


#ifdef CPU_ONLY
STUB_GPU(GradToNormLayer);
#endif

INSTANTIATE_CLASS(GradToNormLayer);
REGISTER_LAYER_CLASS(GradToNorm);

}  // namespace caffe
