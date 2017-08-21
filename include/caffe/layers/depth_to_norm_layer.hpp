#ifndef CAFFE_DEPTH_TO_NORM_LAYER_HPP_
#define CAFFE_DEPTH_TO_NORM_LAYER_HPP__

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * This layer convert the depth map into normal map
 * Including both forward and backward functions
 * bottom[0]: [n, 1, x, x]
 * top[0]: [n, 3, x, x]
 *
 * NOTE: the channel 0 of gradient is -x/z and the channel 1 is -y/z
 * The normal channels are [x, y, z]
 */
template <typename Dtype>
class DepthToNormLayer : public Layer<Dtype> {
 public:
  explicit DepthToNormLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DepthToNorm"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // Normalize the surface normal
  void normalize(Dtype& x, Dtype& y, Dtype& z);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){}

 protected:
  // The radius to calc the normal from depth map
  int radius;

};

}  // namespace caffe

#endif  // CAFFE_DEPTH_TO_NORM_LAYER_HPP_
