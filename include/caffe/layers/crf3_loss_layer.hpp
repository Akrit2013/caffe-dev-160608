#ifndef CAFFE_CRF3_LOSS_LAYER_HPP_
#define CAFFE_CRF3_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

/*
 * This layer is modified from the crf2loaslayer, which use different binary segment
 * result as the pairwise parameter similarity.
 * Instead, this layer use the gpbucm segment result as the pairwise similarity
 * =================================================================================
 * NOTE:
 *	The pairwise term R = alpha * exp(-beta*X)
 *	The X is input of the pairwise value
 *	The beta and the alpha are learnable params
 *	alpha stored in the this->blobs_[0]
 *	beta stored in the this->blobs_[1] 
 * =================================================================================
 * Note this layer perform the CRF loss based on the network output
 * It contains three bottoms
 * bottom[0]: The prediction of the network, which can be n*c*h*w, and the
 * space dimension doesn't matter
 * bottom[1]: The ground truth label, which can be n*c*h*w, also the space dimension
 * will not be used. It only required that the bottom[0]->count() == bottom[1]->count()
 * bottom[2]: The pairwise potential, each channel indicate one type of similarity measurement
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
class Crf3LossLayer : public LossLayer<Dtype> {
 public:
  explicit Crf3LossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param){}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline const char* type() const { return "Crf3Loss"; }
  /**
   * Unlike most loss layers, in the Crf3LossLayer we can backpropagate
   * to both inputs -- override to return true and always allow force_backward.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  /// @copydoc Crf3LossLayer
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

  // Normalize the weight according to the setting
  virtual void Normalize_weight_cpu(void);

  // Calc the loss between Pred_ and the bottom[1]
  virtual void Euclidean_loss_cpu(const Blob<Dtype>* gt, Blob<Dtype>* top);
  virtual void Euclidean_loss_gpu(const Blob<Dtype>* gt, Blob<Dtype>* top);

  // Calc the J matrix for BP process according to the pairwise input bottom[2]
  virtual void Calc_J_cpu(const Blob<Dtype>* bottom);
  virtual void Calc_J_gpu(const Blob<Dtype>* bottom);

  // Do the pairwise bp, and store the diff in the blob_[0].diff
  virtual void Pairwise_BP_cpu(const Blob<Dtype>* gt);
  virtual void Pairwise_BP_gpu(const Blob<Dtype>* gt);

  // Define the type of the unary potential
  enum UnaryDist {
	L2
  };

  UnaryDist unary_mode_;
  // Define the base learning rate for pairwise params
  Dtype pairwise_lr_;

  // Define the internal param
  Blob<Dtype> A_;
  // Define the matrix to store the pairwise relationship
  Blob<Dtype> R_;
  // Define the inverse of the A
  Blob<Dtype> A_inv_;
  // Define the prediction of the crf
  Blob<Dtype> Pred_;
  // Define the J matrix used for pairwise BP
  Blob<Dtype> J_;

  // Params for the pairwise parameters
  // whether to normalize the parameters
  bool normalize_;
  // Whether to keep the parameters positive
  bool positive_;
  // Whether to use partition function
  bool partition_;
  // The interval iteration to display the CRF W parameters, for debug
  // 0 indicate not display
  int disp_w_;
  // The counter to support the disp_w_
  int counter_;
};

}  // namespace caffe

#endif  // CAFFE_CRF3_LOSS_LAYER_HPP_
