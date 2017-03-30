#ifndef CAFFE_SUPERPIXEL_POOLING_LAYER_HPP_
#define CAFFE_SUPERPIXEL_POOLING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layer.hpp"

/*
 * This layer pooling the feature map according to the superpixel map
 * bottom[0] is the feature map [n, c, h, w]
 * bottom[1] is the superpixel map [n, 1, H, W]
 * The output is the pooled feature map: [n, c, Hp, 1], the Hp is the number of
 * superpixels
 */
namespace caffe {

template <typename Dtype>
class SuperpixelPoolingLayer : public Layer<Dtype> {
 public:
  explicit SuperpixelPoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param){}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline const char* type() const { return "SuperpixelPooling"; }

  enum PoolMethod {
	  MAX,
	  AVE,
	  STOCHASTIC
  };

 protected:
  /// @copydoc SuperpixelPoolingLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  PoolMethod pool_method_;
  // The number of the superpixels
  int num_output_;
  // The blob to store the relationship between bottom[0] and superpixel label
  Blob<Dtype> mask_;
  // The Blob to store the number of predictions each superpixel contains
  // It is used for BP
  Blob<Dtype> accum_;
  // This blob has the same as the top blob, it record that every forward
  // whether the corresponding superpixel have a valid value, since when
  // the area of the superpixel is too small or the resolution of the input
  // is not big enough, it might cause that certain superpixel can not be feed
  // 0 indicate the corresponding superpixel has not been feed
  Blob<Dtype> forward_mark_;
  // The rate between the sp height and in height
  int num_;
  int sp_height_;
  int sp_width_;
  int in_channels_;
  int in_height_;
  int in_width_;
  Dtype h_rate_;
  Dtype w_rate_;

};

}  // namespace caffe

#endif  // CAFFE_SUPERPIXEL_POOLING_LAYER_HPP_
