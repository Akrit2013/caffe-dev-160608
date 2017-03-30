// This program is used in PROJ1603, which use CRF to predict the image depth map.
// This program can generate the pairwise map according to the segment boundary map.
// The output map is a float N x N, the N is the number of the superpixels.
// The input:
//	1. A lmdb file containing the superpixels label map in datum format as the target
//	2. A lmdb file containing the segment boundary map in datum format as the reference
// The output:
//  1. A lmdb file containing the generated pairwise map in datum format
//
// Options:
//  1. The pairwise map can be fully connected or not (TODO)
//  2. The pairwise value can use the max of the boundary or the accumulation of the boundary
//
// Usage:
//   generate_pairwise_map [FLAGS] superpixel_lmdb segmetboundary_lmdb pairwise_lmdb

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

#include "caffe/util/generate_pairwise_map_lib.hpp"
#include "caffe/util/mytools.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(fc, true,
    "When this option is on, create the fully connected pairwise map");
DEFINE_bool(uint8, false,
	"If set, rescale the result into 0-255 and save it in uint8 format to save the disk space");
DEFINE_bool(mirror, false,
	"If set, it will mirror the segment boundary map (only segment boundary map, not superpixel map");

DEFINE_string(mode, "max",
        "The mode {max, accum} for calc the pairwise value");

DEFINE_int32(dim, 0, "To explicit set the height and width of the pairwise map");
DEFINE_int32(gpu, 0, "Select the target GPU ID");

DEFINE_string(backend, "lmdb",
        "The backend {leveldb, lmdb} containing the images");

// Parse GPU ids or use all available devices
void get_gpus_id(vector<int>* gpus) {
    int count = 0;
#ifndef CPU_ONLY
    CUDA_CHECK(cudaGetDeviceCount(&count));
#else
    NO_GPU;
#endif
    for (int i = 0; i < count; ++i) {
      gpus->push_back(i);
    }
}

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Create a pairwise lmdb according to the superpixels lmdb \
and the segment boundary lmdb.\n"
        "Usage:\n"
        "    generate_pairwise_map [FLAGS] SUPERPIXEL_LMDB SEG_BOUNDARY_LMDB OUT_PAIRWISE_LMDB");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "generate_pairwise_map");
    return 1;
  }

  // Parse the param
  // const bool is_fc = FLAGS_fc;
  const int dim = FLAGS_dim;
  const int gpu = FLAGS_gpu;
  const string mode = FLAGS_mode;
  const bool use_uint8 = FLAGS_uint8;
  const bool mirror = FLAGS_mirror;

  int pairwise_dim = 0;

  // Init the GPU
  vector<int> gpus;
  get_gpus_id(&gpus);
  cudaDeviceProp device_prop;
  for (int i = 0; i < gpus.size(); ++i) {
	  cudaGetDeviceProperties(&device_prop, gpus[i]);
	  LOG(INFO) << "GPU " << gpus[i] << ": " << device_prop.name;
  }
  Caffe::SetDevice(gpus[gpu]);
  Caffe::set_mode(Caffe::GPU);

  // Open the superpixel and segment lmdb
  LOG(INFO) << "Open " << argv[1] << " database.";
  scoped_ptr<db::DB> db_sup(db::GetDB(FLAGS_backend));
  db_sup->Open(argv[1], db::READ);
  scoped_ptr<db::Cursor> cursor_sup(db_sup->NewCursor());

  LOG(INFO) << "Open " << argv[2] << " database.";
  scoped_ptr<db::DB> db_seg(db::GetDB(FLAGS_backend));
  db_seg->Open(argv[2], db::READ);
  scoped_ptr<db::Cursor> cursor_seg(db_seg->NewCursor());

  // Open the pairwise db
  LOG(INFO) << "Create " << argv[3] << " database.";
  scoped_ptr<db::DB> db_out(db::GetDB(FLAGS_backend));
  db_out->Open(argv[3], db::NEW);
  scoped_ptr<db::Transaction> txn_out(db_out->NewTransaction());

  // Load the first datum and check the data validation
  Datum datum_sup;
  Datum datum_seg;
  datum_sup.ParseFromString(cursor_sup->value());
  datum_seg.ParseFromString(cursor_seg->value());

  // Get the max value of the superpixel map
  pairwise_dim = GetDatumMaxVal(datum_sup) + 1;
  LOG(INFO) << "The max value of the superpixel label: "<<pairwise_dim - 1;
  if (dim > 0){
	  if (dim != pairwise_dim){
		  LOG(WARNING) << "WARNING: The max label and the dim of the pairwise map \
does not match: " <<pairwise_dim<<" vs "<<dim;
	  }
	  pairwise_dim = dim;
  }

  LOG(INFO) << "The shape of the superpixel map: channels "<<datum_sup.channels()<<" height "<<datum_sup.height()<<" width "<<datum_sup.width();
  LOG(INFO) << "The shape of the seg map: channels "<<datum_seg.channels()<<" height "<<datum_seg.height()<<" width "<<datum_seg.width();

  // Start to iter the whole dataset
  int counter = 0;
  while (cursor_sup->valid() && cursor_seg->valid()){
	  Datum datum_seg;
	  Datum datum_sup;
	  Datum datum_out;

	  string key_sup = cursor_sup->key();
	  string key_seg = cursor_seg->key();
	  if (key_sup != key_seg){
		  LOG(ERROR) << "ERROR: The key of the superpixel and segment not match with each other " << key_sup << " VS "<<key_seg;
		  return 1;
	  }

	  datum_seg.ParseFromString(cursor_seg->value());
	  datum_sup.ParseFromString(cursor_sup->value());

	  // Convert the datum to blob
	  Blob<float> blob_seg;
	  Blob<float> blob_sup;
	  Blob<float> blob_out;

	  DatumToBlob(datum_seg, blob_seg, mirror);
	  DatumToBlob(datum_sup, blob_sup);

	  // Generate the pairwise map according to the superpixel and segment
	  blob_out.Reshape(1, 1, pairwise_dim, pairwise_dim);
	  GenPairwiseMap_gpu(blob_sup, blob_seg, blob_out);

	  if (use_uint8){
		  // Rescale the data to 0-255 and save it as uint8_t format to save space
		  blob_out.scale_data(255.0);
	  }
	  // Write the Pairwise map to blob
	  BlobToDatum(blob_out, datum_out, use_uint8);

	  string out;
	  CHECK(datum_out.SerializeToString(&out));
	  txn_out->Put(key_sup, out);

	  counter++;
	  if(counter % 100 == 0){
		  LOG(INFO) << "Processed " << counter << " files.";
		  txn_out->Commit();
		  txn_out.reset(db_out->NewTransaction());
	  }
	  cursor_seg->Next();
	  cursor_sup->Next();
  }
  // write the last batch
  if (counter % 100 != 0) {
    txn_out->Commit();
    LOG(INFO) << "Processed " << counter << " files.";
  }
  return 0;
}
