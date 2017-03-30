#ifndef CAFFE_SUPERPIXEL_CENTROID_LAYER_HPP_
#define CAFFE_SUPERPIXEL_CENTROID_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layer.hpp"

/*
 * This layer find the central coordinates of each superpixels in the image
 * It take the superpixel map as the bottom, and the output shape is:
 * [n, 1, m, 2], m is the number of the suerpixels, and the order of the
 * centroid list m is the same as the superpixel ID in bottom[0] superpixel map
 *
 * The storage of the centroid coordinate is the [height, width]
 */
namespace caffe {

template <typename Dtype>
class SuperpixelCentroidLayer : public Layer<Dtype> {
 public:
  explicit SuperpixelCentroidLayer(const LayerParameter& param)
      : Layer<Dtype>(param){}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline const char* type() const { return "SuperpixelCentroid"; }

 protected:
  /// @copydoc SuperpixelCentroidLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){};
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){};

  // The number of the superpixels
  int num_output_;
  // If set true, the max superpixel id + 1 must equals to the num_output
  // else, the iteartion will stop
  bool check_;
  // If set true, the output coordinate will be normalized according to the
  // width and height of the image
  bool normalize_;
  // The height of the image
  Dtype height_;
  // The width of the image
  Dtype width_;
  // This Blob store the accumulation of the pixels coordinate in each superpixel
  // which is also [n, 1, m, 2], the same as the top[0] blob
  Blob<Dtype> sp_accum_;
  // This Blob store the number of the pixels in each superpixel
  Blob<Dtype> sp_num_;
};

}  // namespace caffe

#endif  // CAFFE_SUPERPIXEL_CENTROID_LAYER_HPP_
