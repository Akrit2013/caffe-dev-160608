#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/depth_to_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void DepthToNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Current, it contains no param
  // Check the top and bottom
  CHECK_EQ(bottom[0]->channels(), 1) <<"The bottom should be a depth map with 1 channels";

  DepthToNormParameter depth2norm_param = this->layer_param_.depth2norm_param();

  if (depth2norm_param.has_radius()){
	  radius = depth2norm_param.radius();
  }else{
	  radius = 2;
  }

}

template <typename Dtype>
void DepthToNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  top[0]->Reshape(bottom[0]->num(), 3, bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void DepthToNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int stride = bottom[0]->count(2);
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  int r = radius;

  // Iter the pixels
  for (int n = 0; n < bottom[0]->num(); n++){
	  for (int h = 0; h < bottom[0]->height(); h++){
		  for (int w = 0; w < bottom[0]->width(); w++){
			  const int bottom_idx = bottom[0]->offset(n, 0, h, w);
			  const int top_idx = top[0]->offset(n, 0, h, w);
			  const int wl = max(0, w - radius);
			  const int wr = min(width - 1, w + radius);
			  const int hu = max(0, h - radius);
			  const int hd = min(height - 1, h + radius);
			  const Dtype ws = wr - wl;
			  const Dtype hs = hd - hu;
			  const int bottom_idx_wl = bottom[0]->offset(n, 0, h, wl);
			  const int bottom_idx_wr = bottom[0]->offset(n, 0, h, wr);
			  const int bottom_idx_hu = bottom[0]->offset(n, 0, hu, w);
			  const int bottom_idx_hd = bottom[0]->offset(n, 0, hd, w);
			  Dtype dx = (bottom_data[bottom_idx_wl] - bottom_data[bottom_idx_wr]) / ws;
			  Dtype dy = (bottom_data[bottom_idx_hu] - bottom_data[bottom_idx_hd]) / hs;
			  Dtype dz = 1;
			  normalize(dx, dy, dz);

			  top_data[top_idx] = dx;
			  top_data[top_idx + stride] = dy;
			  top_data[top_idx + stride * 2] = dz;
		  }
	  }
  }

}

template <typename Dtype>
void DepthToNormLayer<Dtype>::normalize(Dtype& x, Dtype& y, Dtype& z){
	Dtype len = sqrt(x*x + y*y + z*z);
	x = x / len;
	y = y / len;
	z = z / len;
	return;
}

#ifdef CPU_ONLY
STUB_GPU(DepthToNormLayer);
#endif

INSTANTIATE_CLASS(DepthToNormLayer);
REGISTER_LAYER_CLASS(DepthToNorm);

}  // namespace caffe
