#ifndef CAFFE_DISTSQ_CORR_MAT_LAYER_HPP_
#define CAFFE_DISTSQ_CORR_MAT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layer.hpp"

/*
 * This layer calc the correlation matrix of the distance square between superpixels
 * The bottom layer is the [n, 1, m, 2] which produced by the SuperpixelCentroidLayer
 * m is the number of the superpixels 
 * The storage of the centroid coordinate is the [height, width]
 * The output of the layer is [n, 1, m, m].
 */
namespace caffe {

template <typename Dtype>
class DistSqCorrMatLayer : public Layer<Dtype> {
 public:
  explicit DistSqCorrMatLayer(const LayerParameter& param)
      : Layer<Dtype>(param){}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline const char* type() const { return "DistSqCorrMat"; }

 protected:
  /// @copydoc DistSqCorrMatLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){};
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){};

  // If need to normalize the distance according to the width of the image
  bool normalize_;
  // The width of the image to normalize the distance
  int width_;
  // The height of the image to normalize the distance
  int height_;

 protected:
  // The protected functions
  // Calc the square distance between two superpixels
  // This function will normalize the dist if needed
  Dtype CalcDistanceSquare(const Dtype* pCentroid1, const Dtype* pCentroid2);
};

}  // namespace caffe

#endif  // CAFFE_DIST_CORR_MAT_LAYER_HPP_
