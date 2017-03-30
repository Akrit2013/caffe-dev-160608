// This program is used in PROJ1603, it can merge the orginal superpixel map lmdb
// and the mirror superpixel map into a new 2 channel superpixel map,
// which use the channel 0 to store the original superpixel map, and use the channel 1
// to store the mirror superpixel map 
//
// The input:
//  1. The lmdb containing the orginal superpixel map.
//  2. The lmdb containing the mirror superpixel map
// The output:
//  1. The packed superpixel map
//
// Usage:
//   merge_mirror_superpixel_map [FLAGS] original_superpixel_lmdb mirror_superpixel_lmdb out_lmdb

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

/*
DEFINE_bool(fc, true,
    "When this option is on, create the fully connected pairwise map");
DEFINE_bool(mirror, false,
	"If set, it will mirror the segment boundary map (only segment boundary map, not superpixel map");

DEFINE_string(mode, "max",
        "The mode {max, accum} for calc the pairwise value");

DEFINE_int32(dim, 0, "To explicit set the height and width of the pairwise map");
DEFINE_int32(gpu, 0, "Select the target GPU ID");

		*/

DEFINE_bool(uint8, false,
	"If set, rescale the result into 0-255 and save it in uint8 format to save the disk space");
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

  gflags::SetUsageMessage("Create a merged pairwise lmdb containing both original \
and the mirror pairwise map.\n"
        "Usage:\n"
        "    merge_mirror_superpixel_map [FLAGS] PAIRWISE_LMDB MIRROR_PAIRWISE_LMDB OUT_MERGED_LMDB");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "merge_mirror_superpixel_map");
    return 1;
  }

  const bool use_uint8 = FLAGS_uint8;

  // Open the superpixel and segment lmdb
  LOG(INFO) << "Open " << argv[1] << " database.";
  scoped_ptr<db::DB> db_pair(db::GetDB(FLAGS_backend));
  db_pair->Open(argv[1], db::READ);
  scoped_ptr<db::Cursor> cursor_pair(db_pair->NewCursor());

  LOG(INFO) << "Open " << argv[2] << " database.";
  scoped_ptr<db::DB> db_mirror(db::GetDB(FLAGS_backend));
  db_mirror->Open(argv[2], db::READ);
  scoped_ptr<db::Cursor> cursor_mirr(db_mirror->NewCursor());

  // Open the pairwise db
  LOG(INFO) << "Create " << argv[3] << " database.";
  scoped_ptr<db::DB> db_out(db::GetDB(FLAGS_backend));
  db_out->Open(argv[3], db::NEW);
  scoped_ptr<db::Transaction> txn_out(db_out->NewTransaction());

  // Load the first datum and check the data validation
  Datum datum_pair;
  Datum datum_mirr;
  datum_pair.ParseFromString(cursor_pair->value());
  datum_mirr.ParseFromString(cursor_mirr->value());

  LOG(INFO) << "The shape of the original map: channels "<<datum_pair.channels()<<" height "<<datum_pair.height()<<" width "<<datum_pair.width();
  LOG(INFO) << "The shape of the mirror map: channels "<<datum_mirr.channels()<<" height "<<datum_mirr.height()<<" width "<<datum_mirr.width();

  CHECK_EQ(datum_pair.channels(), datum_mirr.channels());
  CHECK_EQ(datum_pair.height(), datum_mirr.height());
  CHECK_EQ(datum_pair.width(), datum_mirr.width());

  // Start to iter the whole dataset
  int counter = 0;
  while (cursor_mirr->valid() && cursor_pair->valid()){
	  Datum datum_pair;
	  Datum datum_mirr;
	  Datum datum_out;

	  string key_pair = cursor_pair->key();
	  string key_mirr = cursor_mirr->key();
	  if (key_pair != key_mirr){
		  LOG(ERROR) << "ERROR: The key of the two candidate lmdb not match with each other " << key_mirr << " VS "<<key_pair;
		  return 1;
	  }

	  datum_pair.ParseFromString(cursor_pair->value());
	  datum_mirr.ParseFromString(cursor_mirr->value());

	  // Convert the datum to blob
	  Blob<float> blob_pair;
	  Blob<float> blob_mirr;
	  Blob<float> blob_out;

	  DatumToBlob(datum_pair, blob_pair);
	  DatumToBlob(datum_mirr, blob_mirr);

	  // Generate the pairwise map according to the superpixel and segment
	  blob_out.Reshape(1, 2, datum_pair.height(), datum_pair.width());

	  // Merge the data
	  const float* pair_data = blob_pair.cpu_data();
	  const float* mirr_data = blob_mirr.cpu_data();
	  float* out_data = blob_out.mutable_cpu_data();

	  for (int n = 0; n < blob_out.num(); n++){
		  for (int h = 0; h < blob_out.height(); h++){
			  for (int w = 0; w < blob_out.width(); w++){
				  out_data[blob_out.offset(n, 0, h, w)] = pair_data[blob_pair.offset(n, 0, h, w)];
				  out_data[blob_out.offset(n, 1, h, w)] = mirr_data[blob_pair.offset(n, 0, h, w)];
			  }
		  }
	  }

	  // Write the Pairwise map to blob
	  BlobToDatum(blob_out, datum_out, use_uint8);

	  string out;
	  CHECK(datum_out.SerializeToString(&out));
	  txn_out->Put(key_pair, out);

	  counter++;
	  if(counter % 100 == 0){
		  LOG(INFO) << "Processed " << counter << " files.";
		  txn_out->Commit();
		  txn_out.reset(db_out->NewTransaction());
	  }
	  cursor_pair->Next();
	  cursor_mirr->Next();
  }
  // write the last batch
  if (counter % 100 != 0) {
    txn_out->Commit();
    LOG(INFO) << "Processed " << counter << " files.";
  }
  return 0;
}
