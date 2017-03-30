#ifndef CAFFE_CRF_NORM_LOSS_LAYER_HPP_
#define CAFFE_CRF_NORM_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

// Added by YanHan
/*
 * This layer is modified from the crf_unary_loss_layer, which is used in PROJ1604 which
 * use predicted normal to guide the CRF.
 * bottom[0]: [n, 1, Hp, 1]. The superpixel pooled depth prediction, Hp is the number of the superpixel
 * bottom[1]: [n, 1, Hp, 1]. The superpixel pooled depth label
 * bottom[2]: [n, 3(2), Hp, 1]. The superpixel pooled normal prediction or gradiant prediction
 * bottom[3]: [n, 1, Hp, 2]. The centroid coordination provided by the SuperpixelCentroidLayer
 *		The 2 elements on width is [height, width] coordinate.
 * TODO:
 * bottom[4]: [n, 1, Hp, Hp]. The superpixel appearance CorrMat provided by the DistCosCorrMatLayer [optional]
 *
 *
 * top[0]: [n, 1, 1, 1]. The loss value
 * top[1]: [n, 1, Hp, 1]. The optimizad depth prediciton [optional]*
 *
 * NOTE:
 * The energy can be calculated as:
 * E = YIY - 2ZIY + ZIZ + YDY - YRY -YKY + YMY + YNY
 * A = I + D - R - K + M + N
 * When perform the inference, the pred = (A_inv)Z
 */
namespace caffe {

template <typename Dtype>
class CrfNormLossLayer : public LossLayer<Dtype> {
 public:
  explicit CrfNormLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param){}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual inline int ExactNumBottomBlobs() const { return 4; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline const char* type() const { return "CrfNormLoss"; }
  /**
   * Unlike most loss layers, in the CrfNormLossLayer we can backpropagate
   * to both inputs -- override to return true and always allow force_backward.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  /// @copydoc CrfNormLossLayer
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

  // Calc the D matrix according to the superpixel centroid bottom and f
  virtual void Calc_D_cpu(const Blob<Dtype>*);

  // Calc the G matrix according to the superpixel pooled normal map
  virtual void Calc_G_cpu(const Blob<Dtype>*);
  virtual void Calc_G_gpu(const Blob<Dtype>*);

  // Calc the R matrix according to the pairwise input
  virtual void Calc_R_cpu(const vector<Blob<Dtype>*>&);
  virtual void Calc_RD_gpu(const vector<Blob<Dtype>*>&);

  // In forward process calc the matrix A according to R
  // The param is the pairwise matrix
  // A = I + D - R - K + M + N
  virtual void Calc_A_cpu(void);
  virtual void Calc_A_gpu(void);

  // Calc the A_inv matrix according to A
  virtual void Calc_A_inv_cpu(void);
  virtual void Calc_A_inv_gpu(void);

  // Init the P and Q tmp matrix, only used in ScaleInvariant mode
  virtual void Init_QP_cpu(void);
  virtual void Init_QP_gpu(void);

  // Inference the network
  virtual void Inference_cpu(const Blob<Dtype>* Z);
  virtual void Inference_gpu(const Blob<Dtype>* Z);

  // In scale invariant mode, use this to infer the prediction
  virtual void Inference_scaleinvariant_cpu(const Blob<Dtype>* Z);
  virtual void Inference_scaleinvariant_gpu(const Blob<Dtype>* Z);

  // Calc the loss between Pred_ and the bottom[1]
  virtual void Euclidean_loss_cpu(const Blob<Dtype>* gt, Blob<Dtype>* top);
  virtual void Euclidean_loss_gpu(const Blob<Dtype>* gt, Blob<Dtype>* top);

  // Calc the Berhu loss when in berhu mode
  virtual void Berhu_loss_cpu(const Blob<Dtype>* gt, Blob<Dtype>* top);
  virtual void Berhu_loss_gpu(const Blob<Dtype>* gt, Blob<Dtype>* top);

  // Calc the Loss using scale invariant loss function
  virtual void ScaleInvariant_loss_cpu(const Blob<Dtype>* gt, Blob<Dtype>* top);
  virtual void ScaleInvariant_loss_gpu(const Blob<Dtype>* gt, Blob<Dtype>* top);

  // Calc the projected depth diff between two superpixels
  // (abs(S_i - S_i') + abs(S_j - S_j')) / abs(D_ij)
  // NOTE: The D and G must be calculate before this function
  virtual Dtype Calc_project_depth_diff_cpu(const int n, const int idx1, const int idx2, const Blob<Dtype>* dep);


  // Define the type of the unary potential
  enum UnaryDist {
	L2,
	Berhu,
	ScaleInvariant
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
  // The tmp matrix Q, in scale invariant mode, A' = A-Q
  Blob<Dtype> Q_;
  // The tmo matrix P, P = I - Q
  // In scale invariant mode, the E = YA'Y - 2ZPY + ZPZ
  Blob<Dtype> P_;
  // Tmp blob used in scale invariant loss
  Blob<Dtype> vecSum_;
  // The buffer blob used in scale invariant mode
  Blob<Dtype> buf_;
  // Define the prediction of the crf
  Blob<Dtype> Pred_;

  // Define the distance matrix D, which is a vector from point A to point B
  // and the f will be used to normalize this matrix
  // D(a,b) = D(b) - D(a)
  // The shape of this matrix is [n, 2, sp_num. sp_num], sp_num is the number
  // of the superpixel, and the first channel contains the height, the second
  // channel contains the width
  Blob<Dtype> D_;

  // Define the matrix contains the gradiant of each superpixel
  // The gradiant of superpixel can be obtained from [x, y, z]
  // by [-y/z, -x/z], corresponding to the height and width
  // The shape of the G_ is [n, 2, sp_num, 1]
  // The first channels is -x/z and the second channel is -y/z
  Blob<Dtype> G_;

  // Should the diff be normalized
  bool normalize_;

  // The weight of the pairwise term
  Dtype alpha_;
  // The weight of the normal difference
  Dtype theta_;
  // The weight of scale invariant term for scale invariant loss
  Dtype delta_;
  // The weight of the normal similarity
  Dtype w1_;
  // The weight of the position similarity
  Dtype w2_;
  // The weight of the normal guided depth similarity
  Dtype w3_;

  // The focal length of the images
  Dtype f_;
  // The height of the output depth map
  Dtype height_;
  // The width of the output depth map
  Dtype width_;
  // The number of the superpixel
  int superpixel_num_;

  // The param for berhu
  Dtype	c_rate_;
  Dtype min_label_;
  Dtype max_label_;
  Dtype invalid_label_;
  bool  has_min_label_;
  bool  has_max_label_;
  // Whether has the appearance input bottom to calc the R matrix
  bool  has_appearance_;
  // Whether the surface normal guidance is used.
  // If set to true, this loss function will act as a traditional
  // CRF loss function
  bool disable_normal_guidance_;
  // If adjust the valid pixel number in scale invariant loss
  bool adjust_pixel_num_;
  // If use the gradiant prediction directly as input
  bool use_gradient_;
  
  // A temp blob to count the bad pixel
  Blob<Dtype>	bad_pixel_;
};

}  // namespace caffe

#endif  // CAFFE_CRF_NORM_LOSS_LAYER_HPP_
