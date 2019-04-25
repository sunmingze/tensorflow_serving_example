#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <glog/logging.h>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/stat.h>
#include <sys/timeb.h>

#include "commonCtr.h"
#include "common_feature.h"
#include "dict_load.h"
#include "model.h"
#include "shm_counter.h"

#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"
#include "tensorflow/core/platform/test_benchmark.h"




using namespace tensorflow;




namespace ctr {
  bool ctr::TfModel::loadModel() {
    long long start = ctr::CommonCtr::getSystemTime();
    std::string successFile = this->localModelPath + "/_SUCCESS";

    if (!isFirstInit || !ctr::CommonCtr::checkLocalFileExist(successFile)) {
        // step 1. 更新模型到本地
        uploadModel2Local(this->localModelPath, this->modelPath);
        MI << "[load tf model 2 local] finish " << this->localModelPath << " " << this->modelPath;
    }

    // step 2. 更新本地模型到内存
    if (!ctr::CommonCtr::checkLocalFileExist(successFile)) {
        ME << "[load tf model] error , cannot find _SUCCESS file";
        return false;
    }
    MI << "Is first init : " << isFirstInit << ", model " << this->localModelPath << " is exist";

    //判断本地文件sucess时间戳跟系统保存时间戳是否相同
    struct stat st;
    stat(successFile.c_str(), &st);
    if (this->lastModifyTime == st.st_mtime) {
        MI << "[load tf model] no necessary to update model, _SUCCESS "
              "modifytime is not changed, lastModifyTime=" << this->lastModifyTime;
        return false;
    }

    // 更新  groupId2Index Map
    if (!ctr::TfModel::readIdMap( this->tfData[1 - this->curPos].feaId2IndexMap)) {
      ME << "[load tf model] error , readIdMap feaGroupIndex 2 id ";
      return false;
    }


    if (!tensorflow::MaybeSavedModelDirectory(this->localModelPath)) {
      ME << "[load tf model] error , localModelPath has no model error";
      return false;
    }


    tensorflow::SessionOptions sess_options;
    tensorflow::RunOptions run_options;
    Status status;

    status = tensorflow::LoadSavedModel(sess_options, run_options, this->localModelPath, {tensorflow::kSavedModelTagServe}, &this->tfData[1 - this->curPos].bundle);
    if (!status.ok()) {
        MI << "[load tf model] error " << status.ToString();
        return false;
    } else {
        MI << status.ToString();
    }

    MI << "[load model] cur model pos: " << this->curPos;
    MI << "[load model] load model pos: " << 1 - this->curPos;
    this->lastModifyTime = st.st_mtime;
    MI << "[load model] update model lastModifyTime=" << this->lastModifyTime;
    long long end = CommonCtr::getSystemTime();
    MI << "[load model] load model total time : " << end - start;
    return true;

  }

  float ctr::TfModel::calScoreV2(const std::string &modelId, const std::vector<std::pair<std::string, GroupedFeature>> &features,
                                 std::vector<std::pair<std::string, std::string>> &featureDebug, bool isDebug) {
    return 0.0;
  }

  void ctr::TfModel::uploadModel2Local(const std::string &localModelDir, const std::string &hdfsDir) {
    long long startTime = CommonCtr::getSystemTime();

    std::string cmd = ctr::projectDir + "/script/upload_dnn_tf_model.sh " + localModelDir + " " + hdfsDir + " " + std::to_string(this->lastModifyTime);
    MI << "execute command: " << cmd;
    exec(cmd);
    long long endTime = CommonCtr::getSystemTime();
    MI << "load model spendtime: " << endTime - startTime << ", model dir: " << hdfsDir;
  }

  void ctr::TfModel::calcScoreForTf(std::vector<float> &scoreVec, std::vector<std::vector<std::pair<std::string, GroupedFeature>>> &docFeatureVec,
                                    bool isDebug) {
    long long start_calscore = ctr::CommonCtr::getSystemTime();
    // graph node key ->  Tensor
    std::vector<std::pair<std::string, Tensor>> inputs;

    // structs : 特征组 --> docids' feature ( 二维数组 )
    std::vector<std::vector<std::pair<int64_t, int64_t>>> indices_list;
    indices_list.resize(this->tfData[this->curPos].totalGroupIdSet.size());

    std::vector<std::vector<int64_t>> value_list;
    value_list.resize(this->tfData[this->curPos].totalGroupIdSet.size());

    std::vector<std::pair<int64_t,int64_t>> shape_list;
    shape_list.resize(this->tfData[this->curPos].totalGroupIdSet.size());


    // 构造 vector
    initFeatureTensor(docFeatureVec, indices_list, value_list, shape_list);

    int group_index = 0;
    for (const auto featureGroupBaseId: this->tfData[this->curPos].totalGroupIdSet) {
      auto indices_name = std::to_string(featureGroupBaseId) + "_indice:0";
      auto value_name = std::to_string(featureGroupBaseId) + "_value:0";
      auto shape_name = std::to_string(featureGroupBaseId) + "_shape:0";

      // 最低也得有15条
      // MI << " indices_vector_list [ group_index ] : " << featureGroupBaseId << " : size "  << indices_list[group_index].size();

      //init indices  2个一对，
      uint64_t ndim = indices_list[group_index].size();
      tensorflow::Tensor tensor_indices(tensorflow::DT_INT64, tensorflow::TensorShape({ndim, 2}));
      auto indice_tensor_map = tensor_indices.tensor<int64, 2>();
      for (int b = 0; b < ndim; b++) {
          indice_tensor_map(b, 0) = indices_list[group_index][b].first;
          indice_tensor_map(b, 1) = indices_list[group_index][b].second;
      }

      // init value
      tensorflow::Tensor tensor_value(tensorflow::DT_INT64, tensorflow::TensorShape({ndim}));
      for (int b = 0; b < ndim ; b++) {
        tensor_value.vec<int64>()(b) = value_list[group_index][b];
      }

      // init shape
      tensorflow::Tensor tensor_shape(tensorflow::DT_INT64, tensorflow::TensorShape({2}));
      tensor_value.vec<int64>()(0) = shape_list[group_index].first;
      tensor_value.vec<int64>()(1) = shape_list[group_index].second;

      // place holder dict
      inputs.emplace_back(std::make_pair(indices_name, tensor_indices));
      inputs.emplace_back(std::make_pair(value_name, tensor_value));
      inputs.emplace_back(std::make_pair(shape_name, tensor_shape));

      group_index++;
    }


    long long end_prepare_tensor = ctr::CommonCtr::getSystemTime();
    falcon::ShmCounter::instance()->Set("tf-prepare-tensor-cost", end_prepare_tensor - start_calscore, falcon::kPercentGauge);
    // MI << "[ prepare-tensor-cost ] : " << end_prepare_tensor - start_calscore;

    std::string output_node_name = "Sigmoid:0";

    std::vector<tensorflow::Tensor> outputs;
    std::unique_ptr<tensorflow::Session> &session = this->tfData[this->curPos].bundle.session;
    Status status;
    //long long start_run = ctr::CommonCtr::getSystemTime();
    status = session->Run(inputs, {output_node_name}, {}, &outputs);

    // MI << "[ tf-session-run-cost ] : " << ctr::CommonCtr::getSystemTime() - start_run;

    if (!status.ok()) {
      MI << status.ToString();
    }

    auto vec = outputs[0].tensor<float, 2>();
    for (int i = 0; i < docFeatureVec.size(); i++) {
      scoreVec[i] = vec(0, i);
     //   MI << "scoreVec[" << i << "]" << scoreVec[i];
    }
    falcon::ShmCounter::instance()->Set("tf-session-score-cost", ctr::CommonCtr::getSystemTime() - end_prepare_tensor, falcon::kPercentGauge);

  }

  void ctr::TfModel::updateModelPtr() {
    MI << "[load model] updating model map";
    MI << "[load model] curPos before update: " << this->curPos;
    this->curPos = 1 - this->curPos;
    isFirstInit = false;
    MI << "[load model] curPos after update: " << this->curPos;
  }


  bool ctr::TfModel::readIdMap(nark::easy_use_hash_map<int64_t,uint64_t> &feaId2IndexMap) {
    feaId2IndexMap.clear();
    uint64_t fea_index = 0;
    for (const auto featureGroupBaseId: this->tfData[this->curPos].totalGroupIdSet) {
      feaId2IndexMap.emplace(featureGroupBaseId, fea_index);
      fea_index++;
    }
    return true;
  };


  void ctr::TfModel::initFeatureTensor(std::vector<std::vector<std::pair<std::string, GroupedFeature>>> &docFeatureVec,
                                       std::vector<std::vector<std::pair<int64_t, int64_t>>> &indices_list,
                                       std::vector<std::vector<int64_t>> &value_list,
                                       std::vector<std::pair<int64_t,int64_t>> &shape_list) {

    nark::easy_use_hash_map<int64_t, uint64_t> &curIdMap = this->tfData[this->curPos].feaId2IndexMap;
    nark::easy_use_hash_map<int64_t, uint64_t>::const_iterator it = curIdMap.end();

    std::vector<int64_t> max_fea;
    max_fea.resize(this->tfData[this->curPos].totalGroupIdSet.size());
    uint64_t group_index;

    for (int64_t i = 0; i < docFeatureVec.size(); i++) {
      auto &features = docFeatureVec[i];

      // feature group in request
      for (auto &featurePair : features) {
        const auto &featureGroup = featurePair.second;
        const auto &featureVec = featureGroup.features;
        // find feature group index by group id

        it = curIdMap.find(featureGroup.group.id);
        if (it != curIdMap.end()){
          group_index = it->second;
        }
        else {
         // DMI << " no contains " << featureGroup.group.name << " id " << featureGroup.group.id;
          continue;
        }

        if (featureVec.size() == 0) {
          indices_list[group_index].emplace_back(std::make_pair(i,0));
          value_list[group_index].emplace_back(0);
          continue;
        }

        // init values
        for (int64_t j = 0; j < featureVec.size(); j++) {
          indices_list[group_index].emplace_back(std::make_pair(i,j));
          value_list[group_index].emplace_back(featureVec[j].identifier);
        }
        // update shape size
        max_fea[group_index] = value_list[group_index].size() > max_fea[group_index] ? value_list[group_index].size()
                                                                                     : max_fea[group_index];
      }
    }


    for (int i = 0; i < max_fea.size(); i++) {
      shape_list[i].first = docFeatureVec.size();
      shape_list[i].second = max_fea[i];
    }
  }

}
