
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include "util.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"
#include "tensorflow/core/platform/test_benchmark.h"


using namespace std;
using namespace tensorflow;


void init_feature(std::vector<std::pair<std::string, Tensor>> &inputs) {
  Feature feature;
  std::string config_fea_name = "2,3,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,30,31,47,48,49,51,90,93,94,95,97,101,113,115,119,120,121,123,124,126,131,132,133,134,135,"
          "136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155";


  std::vector<std::string> config_fea_field;
  util::split(config_fea_name, ',', config_fea_field);


  for (auto i = 0; i < config_fea_field.size(); i++) {
    // init sparse tensor
    auto indices_name = config_fea_field[i] + "_indice:0";
    // 2个一对，一共几对非空
    auto tensor_indices = test::AsTensor<int64>({0, 0, 0, 1, 0, 2, 1, 0, 1, 1, 1, 2}, {6, 2});
    inputs.push_back(std::make_pair(indices_name, tensor_indices));


    auto value_name = config_fea_field[i] + "_value:0";
    auto tensor_value = test::AsTensor<int64>({0, 1, 2, 0, 1, 2});
    inputs.push_back(std::make_pair(value_name, tensor_value));

    // 样本行数  &&  样本中最大 id 量
    auto shape_name = config_fea_field[i] + "_shape:0";
    auto tensor_shape = test::AsTensor<int64>({2, 3});
    inputs.push_back(std::make_pair(shape_name, tensor_shape));
  }

}


int main(int argc, char *argv[]) {


  string modelpath;

  if (argc < 2) {
    cout << "请输入模型路径";

    return 0;
  } else {
    modelpath = argv[1];
  }

  tensorflow::SessionOptions sess_options;
  tensorflow::RunOptions run_options;
  tensorflow::SavedModelBundle bundle;
  Status status;
  status = tensorflow::LoadSavedModel(sess_options, run_options, modelpath, {tensorflow::kSavedModelTagServe}, &bundle);

  if (!status.ok()) {
    cout << status.ToString() << endl;
  }

  tensorflow::MetaGraphDef graph_def = bundle.meta_graph_def;
  std::unique_ptr<tensorflow::Session> &session = bundle.session;


  std::string output_node_name = "Sigmoid:0";


  // 构造接入 placeholder 的 sparse tensor
  std::vector<std::pair<std::string, Tensor>> inputs;
  init_feature(inputs);


  std::vector<tensorflow::Tensor> outputs;
  status = session->Run(inputs, {output_node_name}, {}, &outputs);
  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    return 1;
  } else {
    std::cout << "Run session successfully" << std::endl;
  }


  // Print the results
  std::cout << outputs[0].DebugString() << std::endl;

  for (int i = 0; i <= 1; ++i) {
    std::cout << outputs[0].matrix<float>()(0, i) << " ";
  }

  // Free any resources used by the session
  session->Close();


  return 0;

}