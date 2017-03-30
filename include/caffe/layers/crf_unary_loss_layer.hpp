#ifndef CAFFE_CRF_UNARY_LOSS_LAYER_HPP_
#define CAFFE_CRF_UNARY_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

// Added by YanHan
/*
 * This is a CRF loss layer only contains the unary term, the pairwise term will be provided
 * by the PairwiseMapLayer
 * bottom[0]: The superpixel pooled prediction
 * bottom[1]: The superpixel pooled ground truth
 * bottom[2]: The R matrix provided by PairwiseMapLayer
 *
 * This layer is modified from the crf3_loss_layer, which can be used for multi
 * channels prediciton.
 * The difference is, this layer dose not perform the learning for pariwise params
 * It directly take the R as the pariwise input
 */
namespace caffe {

template <typename Dtype>
class CrfUnaryLossLayer : public LossLayer<Dtype> {
 public:
  explicit CrfUnaryLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param){}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline const char* type() const { return "CrfUnaryLoss"; }
  /**
   * Unlike most loss layers, in the CrfUnaryLossLayer we can backpropagate
   * to both inputs -- override to return true and always allow force_backward.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  /// @copydoc CrfUnaryLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // Calc the R matrix according to the pairwise input
  virtual void Calc_R_cpu(const Blob<Dtype>*);
  virtual void Calc_R_gpu(const Blob<Dtype>*);

  // In forward process calc the matrix A according to R
  // The param is the pairwise matrix
  virtual void Calc_A_cpu(void);
  virtual void Calc_A_gpu(void);

  // Calc the A_inv matrix according to A
  virtual void Calc_A_inv_cpu(void);
  virtual void Calc_A_inv_gpu(void);

  // Inference the network
  virtual void Inference_cpu(const Blob<Dtype>* Z);
  virtual void Inference_gpu(const Blob<Dtype>* Z);

  // Calc the loss between Pred_ and the bottom[1]
  virtual void Euclidean_loss_cpu(const Blob<Dtype>* gt, Blob<Dtype>* top);
  virtual void Euclidean_loss_gpu(const Blob<Dtype>* gt, Blob<Dtype>* top);

  // Calc the Berhu loss when in berhu mode
  virtual void Berhu_loss_cpu(const Blob<Dtype>* gt, Blob<Dtype>* top);
  virtual void Berhu_loss_gpu(const Blob<Dtype>* gt, Blob<Dtype>* top);

  // Define the type of the unary potential
  enum UnaryDist {
	L2,
	Berhu,
	Berhuber
  };

  // Define the mode to use the c_rate
  enum CRateMode {
	  MAX,
	  AVE
  };

  UnaryDist unary_mode_;

  CRateMode c_rate_mode_;

  // Define the internal param
  Blob<Dtype> A_;
  // Define the matrix to store the pairwise relationship
  Blob<Dtype> R_;
  // Define the inverse of the A
  Blob<Dtype> A_inv_;
  // Define the prediction of the crf
  Blob<Dtype> Pred_;
  // Should the diff be normalized
  bool normalize_;

  // The weight of the R
  Dtype alpha_;

  // The param for berhu
  Dtype	c_rate_;
  Dtype h_rate_;
  Dtype min_label_;
  Dtype max_label_;
  Dtype invalid_label_;
  bool  has_min_label_;
  bool  has_max_label_;
  bool	has_invalid_label_;
  bool	has_h_rate_;
  
  // A temp blob to count the bad pixel
  Blob<Dtype>	bad_pixel_;
};

}  // namespace caffe

#endif  // CAFFE_CRF_UNARY_LOSS_LAYER_HPP_
