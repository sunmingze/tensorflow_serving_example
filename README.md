tensorflow-serving-cpp-example
========

TensorFlow prediction using its C++ API.
Having this repo, you will not need `TensorFlow-Serving`. 

## 1) Build Tensoflow 
* tensorflow 版本 1.12.0 
* bazel 版本 > 0.19.0

Follow the instruction [build tensorflow from source](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/makefile)
```bash
git clone --recursive https://github.com/tensorflow/tensorflow.git
cd tensorflow
sh tensorflow/contrib/makefile/build_all_linux.sh (works for linux and osx)

./configure
bazel build --config=opt //tensorflow:libtensorflow_cc.so

```


## 2) import deps 
```bash

tf_base="/home/work/sunmingze/tf-serving/tensorflow"
lib_path="lib"
protobuf_path="/home/work/sunmingze/tf-serving/proto3.6.1"
eigen_path="/home/work/sunmingze/tf-serving/eigin"
gtest_path="/home/work/sunmingze/tf-serving/gtest"

# import so 
cp $tf_base/bazel-bin/tensorflow/libtensorflow_cc.so $lib_path/
cp $tf_base/bazel-bin/tensorflow/libtensorflow_framework.so $lib_path/
cp $tf_base/tensorflow/contrib/makefile/gen/protobuf/lib/libprotobuf.a $lib_path/
cp $protobuf_path/libprotobuf.a $lib_path/
cp $gest_path/libgtest.a $lib_path/


# copy include
deps_path="deps/tensorflow"
rm -rvf deps/*
mkdir -p $deps_path

cp -r $tf_base/bazel-genfiles/* $deps_path/
cp -r $tf_base/tensorflow/cc $deps_path/tensorflow/
cp -r $tf_base/tensorflow/core $deps_path/tensorflow/
cp -r $tf_base/third_party $deps_path/
cp -r $protobuf_path/include/* $deps_path/
cp -r $eigen_path/include/eigen3/* $deps_path/
cp -r $gtest_path/include/* $deps_path/
cp -r ${tf_base}/tensorflow/contrib/makefile/downloads/absl/absl $deps_path/

mkdir -p $deps_path/external/nsync
cp -r ${tf_base}/tensorflow/contrib/makefile/downloads/nsync/public $deps_path/external/nsync/public


# 删除不需要的cc文件
cd $deps_path
#find . -name "*.cc" -type f -delete




```


## 3) build this repo
```bash
git clone 
cd tensorflow-service-example
cmake ..
make 

```

## 4) save model by python 



