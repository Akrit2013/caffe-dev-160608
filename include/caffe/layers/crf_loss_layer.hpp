#ifndef CAFFE_CRF_LOSS_LAYER_HPP_
#define CAFFE_CRF_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

/*
 * Note this layer perform the CRF loss based on the network output
 * It contains three bottoms
 * bottom[0]: The prediction of the network, which can be n*c*h*w, and the
 * space dimension doesn't matter
 * bottom[1]: The ground truth label, which can be n*c*h*w, also the space dimension
 * will not be used. It only required that the bottom[0]->count() == bottom[1]->count()
 * bottom[2]: The pairwise potential, currently, the channel must be 1, and the
 * height == width == botton[0/1]->channels()*bottom[0/1]->height()*bottom[0/1]->width()
 * The output of this loss is the energy of the CRF
 *
 * In addition, this layer can have 2 tops
 * If only one top is set, it only produce the loss value
 * If the top[1] is also set, it can estimate the depth through the top[1]
 * Its's shape is the same as the bottom[0]
 */
namespace caffe {

template <typename Dtype>
class CrfLossLayer : public LossLayer<Dtype> {
 public:
  explicit CrfLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline const char* type() const { return "CrfLoss"; }
  /**
   * Unlike most loss layers, in the CrfLossLayer we can backpropagate
   * to both inputs -- override to return true and always allow force_backward.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  /// @copydoc CrfLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){};

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){};

  // In forward process calc the matrix A
  // The param is the pairwise matrix
  virtual void Calc_A_cpu(const Blob<Dtype>*);
  virtual void Calc_A_gpu(const Blob<Dtype>*){};
  // This function calc the BP diff value for the pairwise input
  // The return is a Dtype indicate the diff in certain position
  // A tmp blob as buffer can be set to save time for alloc memory
  virtual Dtype BP_r_cpu(const int n, const Blob<Dtype>* A_inv, const Blob<Dtype>* Y, const Blob<Dtype>* Z, const Blob<Dtype>* J, Blob<Dtype>* buf_blob=NULL);
  // This function generate the matrix J, which is used in pairwise BP
  virtual void get_J_cpu(const int h, const int w, Blob<Dtype>* J);
  // Calc the BP of the pairwise
  virtual void BP_pairwise_cpu(const Blob<Dtype>* A_inv, const Blob<Dtype>* Y, const Blob<Dtype>* Z, Blob<Dtype>* Out);
  // Calc the probabilty of the crf
  // It will store the prob of the minibatch into the prob_, also, it will
  // return the mean probabilty of the current minibatch
  virtual Dtype Calc_prob_cpu(const Blob<Dtype>* A, const Blob<Dtype>* Y, const Blob<Dtype>* Z, Blob<Dtype>* A_inv, Blob<Dtype>* prob);
  virtual Dtype Calc_prob_gpu(const Blob<Dtype>* A, const Blob<Dtype>* Y, const Blob<Dtype>* Z, Blob<Dtype>* A_inv, Blob<Dtype>* prob){return 0;};

  // Inference the network
  virtual void Inference_cpu(const Blob<Dtype>* A_inv, const Blob<Dtype>* Z, Blob<Dtype>* Pred);
  virtual void Inference_gpu(const Blob<Dtype>* A_inv, const Blob<Dtype>* Z, Blob<Dtype>* Pred){};

  // Define the type of the unary potential
  enum UnaryDist {
	L2
  };

  UnaryDist unary_mode_;
  // the alpha is the weight of the pairwise part
  Dtype alpha_;
  int channels_;
  // Define the internal param
  Blob<Dtype> A_;
  // Define the inverse of the A
  Blob<Dtype> A_inv_;
  // Define the energy of the loss
  Blob<Dtype> E_;
  // Define the probabilty of the minibatch
  Blob<Dtype> prob_;

  Blob<Dtype> diff_;
};

}  // namespace caffe

#endif  // CAFFE_CRF_LOSS_LAYER_HPP_
