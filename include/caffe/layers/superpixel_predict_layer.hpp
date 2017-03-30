#ifndef CAFFE_SUPERPIXEL_PREDICT_LAYER_HPP_
#define CAFFE_SUPERPIXEL_PREDICT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layer.hpp"

/*
 * This layer take the output of Crf2LossLayer or SuperpixelPoolingLayer
 * as the input, and produce the full size prediction image
 * bottom[0] is the output of the Crf2LossLayer or SuperpixelPoolingLayer, which is
 * the feature map of each super pixel, it should be the [n, c, h, w] or
 * [n, c, output_num, 1]
 * bottom[1] is the superpixel label map which is [n, 1, h1, w1]
 * The top[0] is the output, the size is [n, c, h1, w1]
 */
namespace caffe {

template <typename Dtype>
class SuperpixelPredictLayer : public Layer<Dtype> {
 public:
  explicit SuperpixelPredictLayer(const LayerParameter& param)
      : Layer<Dtype>(param){}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline const char* type() const { return "SuperpixelPredict"; }

 protected:
  /// @copydoc SuperpixelPredictLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
	  for (int i = 0; i < propagate_down.size(); ++i) {
		  if (propagate_down[i]) { NOT_IMPLEMENTED; }
	  }
  }

  // The number of the superpixel
  int sp_num_;

};

}  // namespace caffe

#endif  // CAFFE_SUPERPIXEL_PREDICT_LAYER_HPP_
