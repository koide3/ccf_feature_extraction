#include <iostream>
#include <boost/format.hpp>
#include <opencv2/opencv.hpp>

#include <ros/package.h>
#include <ccf_feature_extraction/ccf_extractor.hpp>

int main(int argc, char** argv) {
  if(argc < 2) {
    std::cout << "usage: ccf_feature_extraction_test image_filename1 image_filename2 ..." << std::endl;
    return 0;
  }

  std::string package_dir = ros::package::getPath("ccf_feature_extraction");

  // The original version of Ahmed's network
  // It has two convolutional layers with 20 and 25 filters, and thus it yields 25 feature maps
  // ccf::AhmedSubnet subnet(package_dir + "/data/cnn_params");

  // A tiny version of Ahmed's network
  // The number of filters of both the convolutional layers are reduced to 10
  ccf::TinyAhmedSubnet subnet(package_dir + "/data/cnn_params_tiny");

  for(int i=1; i<argc; i++) {
    cv::Mat bgr = cv::imread(argv[i]);
    if(!bgr.data) {
      std::cerr << "failed to read " << argv[i] << std::endl;
      continue;
    }

    // The network was trained with RGB pixel order
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, CV_BGR2RGB);

    // Extract features using CCF
    // Each response map has CV_32FC1 pixels
    std::vector<cv::Mat> responses = subnet(rgb);

    cv::Mat all(rgb.rows, rgb.cols * (1 + responses.size()), CV_8UC3);
    rgb.copyTo(cv::Mat(all, cv::Rect(0, 0, rgb.cols, rgb.rows)));

    for(int i=0; i<responses.size(); i++) {
      cv::Mat roi(all, cv::Rect((i + 1) * rgb.cols, 0, rgb.cols, rgb.rows));
      responses[i].convertTo(responses[i], CV_8U, 128.0);
      cv::cvtColor(responses[i], responses[i], CV_GRAY2BGR);
      cv::resize(responses[i], roi, roi.size());
    }

    cv::imshow("responses", all);
    cv::waitKey(0);
  }

  return 0;
}
