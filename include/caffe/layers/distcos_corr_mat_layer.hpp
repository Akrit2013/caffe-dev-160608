#ifndef CAFFE_DISTCOS_CORR_MAT_LAYER_HPP_
#define CAFFE_DISTCOS_CORR_MAT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layer.hpp"

/*
 * This layer is used to calc the cos distqnce between surface normal or appearance
 * vectores of superpixels and generate the correlation matrix.
 *
 * The input bottom[0] should be a superpixel pooled [n, c, m, 1] vector, m is the
 * number of the superpixel.
 * The output top[0] should be a correlation matrix [n, 1, m, m].
 */
namespace caffe {

template <typename Dtype>
class DistCosCorrMatLayer : public Layer<Dtype> {
 public:
  explicit DistCosCorrMatLayer(const LayerParameter& param)
      : Layer<Dtype>(param){}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline const char* type() const { return "DistCosCorrMat"; }

 protected:
  /// @copydoc DistCosCorrMatLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){};
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){};

  // If need to normalize the cosine distance
  bool normalize_;

 protected:
  // The protected functions
  // Calc the cosine distance between two feature vectors
  // This function will normalize the dist if needed
  // The stride is the stride between channels, and the len is the length of the
  // vector
  Dtype CalcCosDistance(const Dtype* data1, const Dtype* data2, const int stride, const int len);
  // Calc the L2Norm of the vector
  Dtype L2Norm(const int count, const int stride, const Dtype* data);
};

}  // namespace caffe

#endif  // CAFFE_DISTCOS_CORR_MAT_LAYER_HPP_
