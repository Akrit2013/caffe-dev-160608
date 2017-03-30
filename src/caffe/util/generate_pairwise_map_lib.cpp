/*
 * This lib contains the functions should be use to generate the pairwise map
 */


#include "caffe/util/generate_pairwise_map_lib.hpp"
#include "caffe/util/mytools.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


/* This function get the max pixel value contained in a datum
 */
float GetDatumMaxVal(Datum& datum){
	// Convert the datum to blob first and then find the largest val
	Blob<float> blob;
	DatumToBlob(datum, blob);
	float val = caffe_amax(blob.count(), blob.cpu_data());
	return val;
}

}
