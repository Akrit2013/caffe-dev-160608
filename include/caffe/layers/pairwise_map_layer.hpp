#ifndef CAFFE_PAIRWISE_MAP_LAYER_HPP_
#define CAFFE_PAIRWISE_MAP_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * This layer take the feature from the superpixel pooling layer
 * which is (n, c, h, 1)
 * The h is the number of the superpixels, each superpixel is a c dim feature
 * vector
 * This layer calc the sumilarity relationship from these feature vectors
 * and product a (n, 1, h, h) blob
 * NOTE: The top[0] is a symmetric matrix
 */
template <typename Dtype>
class PairwiseMapLayer : public Layer<Dtype> {
 public:
  explicit PairwiseMapLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "PairwiseMap"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){}
  /*
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	  */

  // Calc the L2 distance between two superpixels, the input is the
  // index of the superpixels, and the coordinate will be normalized
  // to 0-1 before the distance calculation
  virtual Dtype CalcDistance2(const int index1, const int index2);
  // Calc the L2 norm of the feature vector
  virtual Dtype L2Norm(const int count, const int stride, const Dtype* data);

  // The params
  Dtype theta1_;
  Dtype theta2_;
  Dtype theta3_;

  Dtype w1_;
  Dtype w2_;

  int height_;
  int width_;

};

}  // namespace caffe

#endif  // CAFFE_PAIRWISE_MAP_LAYER_HPP_
