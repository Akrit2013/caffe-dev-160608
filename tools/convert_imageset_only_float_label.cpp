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

  const int BUF_SIZE = 20000;
  char buffer[BUF_SIZE];

  std::vector<std::string> vec_line;
  std::vector<std::string> vec_lines;
  // Store the float vector of each line
  std::vector<std::vector<float> > labels_lines;

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

  // Create new DB
  scoped_ptr<db::DB> db_lab(db::GetDB(FLAGS_backend));
  db_lab->Open(argv[3], db::NEW);
  scoped_ptr<db::Transaction> txn_lab(db_lab->NewTransaction());

  // Storing to db
  std::string root_folder(argv[1]);
  Datum datum_lab;

  int count = 0;

  for (int line_id = 0; line_id < lines.size(); ++line_id) {
	// Read the labels to datum
	FloatVecToDatum(labels_lines[line_id], &datum_lab);

    // sequential
    string key_str = caffe::format_int(line_id, 8) + "_" + lines[line_id];

    // Put in db
	string out_lab;
	CHECK(datum_lab.SerializeToString(&out_lab));
	txn_lab->Put(key_str, out_lab);

    if (++count % 1000 == 0) {
      // Commit db
	  txn_lab->Commit();
	  txn_lab.reset(db_lab->NewTransaction());

      LOG(INFO) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
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
