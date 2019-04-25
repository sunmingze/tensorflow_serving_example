cmake_minimum_required(VERSION 2.8)

include_directories(
        ${PROJECT_SOURCE_DIR}/deps/tensorflow
        ${PROJECT_SOURCE_DIR}/include)


link_directories(${PROJECT_SOURCE_DIR}/lib)


set(DEEP_CTR_LINKER_LIBS "")
list(APPEND DEEP_CTR_LINKER_LIBS
        libprotobuf.a
        libtensorflow_cc.so
        libtensorflow_framework.so)

# executable
set(DEEP_MODEL "deep_model.bin")
add_executable(${DEEP_MODEL} "deep_model.cc")
set(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")
target_link_libraries(${DEEP_MODEL} ${DEEP_CTR_LINKER_LIBS})

