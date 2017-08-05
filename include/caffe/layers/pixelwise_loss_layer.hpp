#ifndef CAFFE_PIXELWISE_LOSS_LAYER_HPP_
#define CAFFE_PIXELWISE_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

// Added by YanHan
/*
 * This layer is used in PROJ1604. It project the superpixel wise depth prediction 
 * into the pixelwise prediction
 *
 * use predicted normal to guide the CRF.
 * bottom[0]: [n, 1, Hp, 1]. The superpixel pooled depth prediction, Hp is the number of the superpixel
 * bottom[1]: [n, 1, H, W]. The depth label
 * bottom[2]: [n, 1, H, W]. The superpixel segment result
 * bottom[3]: [n, 1, Hp, 2]. The centroid coordination provided by the SuperpixelCentroidLayer
 *		The 2 elements on width is [height, width] coordinate.
 * bottom[4][BP]: [n, 3(2), Hp, 1]. The superpixel pooled normal prediction (3 channels)
 *				  or the superpixel pooled gradient on X and Y axes (2 channels)
 *				  NOTE: When take gradient bottom as input, the first channel is [-x/z] corresponding to width
 *				  and hte second channel is [-y/z] corresponding to the height
 *
 * top[0]: [n, 1, 1, 1]. The loss value
 * top[1]: [n, 1, H, W]. The pixelwise prediction result [optional]
 * top[2]: [x, x, x, x]. The top output for debug [optional]
 *
 */
namespace caffe {

template <typename Dtype>
class PixelwiseLossLayer : public LossLayer<Dtype> {
 public:
  explicit PixelwiseLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param){}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline const char* type() const { return "PixelwiseLoss"; }
  /**
   * Unlike most loss layers, in the PixelwiseLossLayer we can backpropagate
   * to both inputs -- override to return true and always allow force_backward.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  /// @copydoc PixelwiseLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // Calc the L2 norm of the feature vector
  virtual Dtype L2Norm(const int count, const int stride, const Dtype* data);
  // Calc the L2 distance square between two superpixels
  virtual Dtype L2DistSq(const Dtype* data1, const Dtype* data2);

  // Calc the loss between Pred_ and the bottom[1]
  virtual void Euclidean_loss_cpu(const Blob<Dtype>* gt, Blob<Dtype>* top);
  virtual void Euclidean_loss_gpu(const Blob<Dtype>* gt, Blob<Dtype>* top);

  // Calc the Berhu loss when in berhu mode
  virtual void Berhu_loss_cpu(const Blob<Dtype>* gt, Blob<Dtype>* top);
  virtual void Berhu_loss_gpu(const Blob<Dtype>* gt, Blob<Dtype>* top);

  // Calc the Loss using scale invariant loss function
  virtual void ScaleInvariant_loss_cpu(const Blob<Dtype>* gt, Blob<Dtype>* top);
  virtual void ScaleInvariant_loss_gpu(const Blob<Dtype>* gt, Blob<Dtype>* top);

  // Inference pixelwise prediction according to the surface normal
  virtual void Pixelwise_inference_cpu(const vector<Blob<Dtype>*>& bottom);
  virtual void Pixelwise_inference_gpu(const vector<Blob<Dtype>*>& bottom);

  // Define the type of the unary potential
  enum LossMode {
	L2,
	Berhu,
	ScaleInvariant
  };
  enum CRateMode {
	  MAX,
	  AVE
  };

  LossMode loss_mode_;
  CRateMode c_rate_mode_;

  // Store the pixel wise depth prediction
  // [n, 1, H, W]
  Blob<Dtype> Pred_;
  // Store the superpixel wise depth diff, which is used for BP
  // [n, 1, superpixel_num_, 1]
  Blob<Dtype> dep_diff_;
  // Store the pixel wise height and width
  Dtype height_;
  Dtype width_;

  // Should the diff be normalized
  bool normalize_;
  // Should BP the depth
  bool bp_depth_;
  // Is the bottom[4] surface normal or the surface gradients
  bool use_gradient_;

  // The weight of scale invariant term for scale invariant loss
  Dtype delta_;

  // The weight of the L2 regulation term
  Dtype lambda_;

  // The focal length of the images
  Dtype f_;
  // The number of the superpixel
  int superpixel_num_;

  // The param for berhu
  Dtype	c_rate_;
  Dtype min_label_;
  Dtype max_label_;
  Dtype z_thd_;
  bool  has_min_label_;
  bool  has_max_label_;

  // If z=fabs(z) during the TEST
  bool force_z_positive_;

  // The learning rate of the z in surface normal bp
  Dtype lr_z_;

  // If the norm diff > mean(diff) * h_rate, it will be set to mean(diff) * h_rate
  Dtype h_rate_;
  // The h_rate_ only be effective when it > 0
  bool has_h_rate_;
  // The pixel wise radius of each superpixel when perform BP step
  Dtype radius_;
  bool has_radius_;
  
  // A temp blob to count the bad pixel
  // Use in cale invariant loss
  Blob<Dtype>	bad_pixel_;
  // The tmp blob used in scale invariant loss
  Blob<Dtype> vecSum_;
  // The blob to store the accumulation of the normal BP
  // The data store the accumlation, and the diff is the counter
  Blob<Dtype> normAccum_;
};

}  // namespace caffe

#endif  // CAFFE_PIXELWISE_LOSS_LAYER_HPP_
