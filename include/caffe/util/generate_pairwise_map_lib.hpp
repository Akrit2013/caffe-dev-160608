/*
 * This lib contains the functions should be use to generate the pairwise map
 */
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"
#include "caffe/common.hpp"
#include "caffe/blob.hpp"

namespace caffe{

float GetDatumMaxVal(Datum& datum);

template <typename Dtype>
void GenPairwiseMap_gpu(const Blob<Dtype>& blob_sup, const Blob<Dtype>& blob_seg, Blob<Dtype>& blob_pair);


}
