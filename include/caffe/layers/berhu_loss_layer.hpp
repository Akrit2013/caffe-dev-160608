#ifndef CAFFE_BERHU_LOSS_LAYER_HPP_
#define CAFFE_BERHU_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {
// Added by YanHan
// This loss layer is a combination of L1 loss and L2 loss

template <typename Dtype>
class BerhuLossLayer : public LossLayer<Dtype> {
 public:
  explicit BerhuLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "EuclideanLoss"; }
  /**
   * Unlike most loss layers, in the BerhuLossLayer we can backpropagate
   * to both inputs -- override to return true and always allow force_backward.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  /// @copydoc BerhuLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_;
  double	c_rate_;
  Dtype		min_label_;
  Dtype		max_label_;
  Dtype		invalid_label_;
  bool		has_min_label_;
  bool		has_max_label_;
  bool		has_invalid_label_;
  bool		normalize_;

  // A temp blob to count the bad pixel
  Blob<Dtype>	bad_pixel_;

  // Deside whether the c_rate is based on max diff or average diff
  enum CRateMode {
	  MAX,
	  AVE
  };

  CRateMode c_rate_mode_;
  Dtype		h_rate_;
  bool		has_h_rate_;
};

}

#endif
