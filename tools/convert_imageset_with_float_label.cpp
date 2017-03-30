// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <cstdlib>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
    "When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
    "Optional: What type should we encode the image as ('png','jpg',...).");

void FloatVecToDatum(const std::vector<float>& lab_vec, Datum* datum);

int main(int argc, char** argv) {
#ifdef USE_OPENCV
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n"
        "The ImageNet dataset for the training demo is at\n"
        "    http://www.image-net.org/download-images\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset");
    return 1;
  }

  const bool is_color = !FLAGS_gray;
  const bool check_size = FLAGS_check_size;
  const bool encoded = FLAGS_encoded;
  const string encode_type = FLAGS_encode_type;

  const int BUF_SIZE = 20000;
  char buffer[BUF_SIZE];

  std::vector<std::string> vec_line;
  std::vector<std::string> vec_lines;
  // Store the float vector of each line
  std::vector<std::vector<float> > labels_lines;

  // The name of the data lmdb
  char dataDB_name[100];
  char labelDB_name[100];
  
  strcpy(dataDB_name, argv[3]);
  strcpy(labelDB_name, argv[3]);

  strcat(dataDB_name, "_data_lmdb");
  strcat(labelDB_name, "_label_lmdb");

  std::ifstream infile(argv[2]);
  // std::vector<std::pair<std::string, int> > lines;
  std::vector<std::string> lines;
  std::string filename;
  // First, load all the lines into a vector
  while(infile){
	  infile.getline(buffer, BUF_SIZE);
	  std::string tmp(buffer);
	  if(tmp.empty()){
		  continue;
	  }
	  vec_lines.push_back(tmp);
	  memset(buffer, 0, BUF_SIZE);
  }


  // Shuffle the lines if needed
  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(vec_lines.begin(), vec_lines.end());
  }

  // Parse the lines, and store the image path and float labels into
  // different vectors
  for(std::vector<std::string>::iterator iter = vec_lines.begin(); iter != vec_lines.end(); iter++){
	  std::string line = *iter;
	  // Split the line
	  boost::split(vec_line, line, boost::is_any_of(" "));
	  // Store the image name in lines
	  lines.push_back(vec_line[0]);
	  // Covert the remain labels into vector<float> and store them into labels_lines
	  std::vector<float> vec_label;
	  for(std::vector<std::string>::iterator iter_lab = vec_line.begin()+1; iter_lab != vec_line.end(); iter_lab++){
		  std::string lab = *iter_lab;
		  // Convert string into float
		  float val = atof(lab.c_str());
		  vec_label.push_back(val);
	  }
	  labels_lines.push_back(vec_label);
  }


  LOG(INFO) << "A total of " << lines.size() << " images.";

  if (encode_type.size() && !encoded)
    LOG(INFO) << "encode_type specified, assuming encoded=true.";

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Create new DB
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  scoped_ptr<db::DB> db_lab(db::GetDB(FLAGS_backend));
  db->Open(dataDB_name, db::NEW);
  db_lab->Open(labelDB_name, db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());
  scoped_ptr<db::Transaction> txn_lab(db_lab->NewTransaction());

  // Storing to db
  std::string root_folder(argv[1]);
  Datum datum;
  Datum datum_lab;

  int count = 0;
  int data_size = 0;
  bool data_size_initialized = false;

  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    bool status;
    std::string enc = encode_type;
    if (encoded && !enc.size()) {
      // Guess the encoding type from the file name
      string fn = lines[line_id];
      size_t p = fn.rfind('.');
      if ( p == fn.npos )
        LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
      enc = fn.substr(p);
      std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
    }
    status = ReadImageToDatum(root_folder + lines[line_id],
        0, resize_height, resize_width, is_color,
        enc, &datum);
	// Read the labels to datum
	FloatVecToDatum(labels_lines[line_id], &datum_lab);

    if (status == false) continue;
    if (check_size) {
      if (!data_size_initialized) {
        data_size = datum.channels() * datum.height() * datum.width();
        data_size_initialized = true;
      } else {
        const std::string& data = datum.data();
        CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
            << data.size();
      }
    }
    // sequential
    string key_str = caffe::format_int(line_id, 8) + "_" + lines[line_id];

    // Put in db
    string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(key_str, out);

	string out_lab;
	CHECK(datum_lab.SerializeToString(&out_lab));
	txn_lab->Put(key_str, out_lab);

    if (++count % 1000 == 0) {
      // Commit db
      txn->Commit();
      txn.reset(db->NewTransaction());

	  txn_lab->Commit();
	  txn_lab.reset(db_lab->NewTransaction());

      LOG(INFO) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    txn->Commit();
	txn_lab->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}

void FloatVecToDatum(const std::vector<float>& lab_vec, Datum* datum) {
	int len = lab_vec.size();
	datum->set_channels(len);
	datum->set_height(1);
	datum->set_width(1);
	datum->clear_data();
	datum->clear_float_data();
	datum->set_encoded(false);
	
	for(std::vector<float>::const_iterator iter = lab_vec.begin(); iter != lab_vec.end(); iter++){
		float val = *iter;
		datum->add_float_data(val);
	}
}
