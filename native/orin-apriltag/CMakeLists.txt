cmake_minimum_required(VERSION 3.28)

project(PhotonOrinAprilTag LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES 87 CACHE STRING "CUDA architecture")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(JNI REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

string(REGEX MATCH "^([0-9]+\\.[0-9]+)(\\.|$)"
    ORIN_CUDA_COMPILER_VERSION_MATCH "${CMAKE_CUDA_COMPILER_VERSION}")
set(ORIN_CUDA_COMPILER_MAJOR_MINOR "${CMAKE_MATCH_1}")
string(REGEX MATCH "^([0-9]+\\.[0-9]+)(\\.|$)"
    ORIN_CUDATOOLKIT_VERSION_MATCH "${CUDAToolkit_VERSION}")
set(ORIN_CUDATOOLKIT_MAJOR_MINOR "${CMAKE_MATCH_1}")
if(NOT ORIN_CUDA_COMPILER_MAJOR_MINOR STREQUAL "13.2" OR
   NOT ORIN_CUDATOOLKIT_MAJOR_MINOR STREQUAL "13.2")
    message(FATAL_ERROR
        "CUDA 13.2 is required for the Orin compatibility shims; observed "
        "CMAKE_CUDA_COMPILER_VERSION='${CMAKE_CUDA_COMPILER_VERSION}' and "
        "CUDAToolkit_VERSION='${CUDAToolkit_VERSION}'.")
endif()

find_path(CUDA_CCCL_INCLUDE_DIR
    cub/device/device_copy.cuh
    HINTS ${CUDAToolkit_INCLUDE_DIRS}
    PATH_SUFFIXES cccl
    REQUIRED)

enable_testing()

if(NOT DEFINED PV_SOURCE_REVISION)
    set(PV_SOURCE_REVISION "unknown")
endif()
string(TIMESTAMP PV_BUILD_TIMESTAMP UTC)
set(PV_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${PV_GENERATED_INCLUDE_DIR}")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/BuildInfo.h.in"
    "${PV_GENERATED_INCLUDE_DIR}/BuildInfo.h"
    @ONLY)

set(APRILTAG_VENDOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/upstream/third_party/apriltag")
set(APRILTAG_VENDOR_SOURCES
    "${APRILTAG_VENDOR_DIR}/apriltag.c"
    "${APRILTAG_VENDOR_DIR}/apriltag_pose.c"
    "${APRILTAG_VENDOR_DIR}/apriltag_quad_thresh.c"
    "${APRILTAG_VENDOR_DIR}/common/g2d.c"
    "${APRILTAG_VENDOR_DIR}/common/getopt.c"
    "${APRILTAG_VENDOR_DIR}/common/homography.c"
    "${APRILTAG_VENDOR_DIR}/common/image_u8.c"
    "${APRILTAG_VENDOR_DIR}/common/image_u8x3.c"
    "${APRILTAG_VENDOR_DIR}/common/image_u8x4.c"
    "${APRILTAG_VENDOR_DIR}/common/matd.c"
    "${APRILTAG_VENDOR_DIR}/common/pam.c"
    "${APRILTAG_VENDOR_DIR}/common/pjpeg.c"
    "${APRILTAG_VENDOR_DIR}/common/pjpeg-idct.c"
    "${APRILTAG_VENDOR_DIR}/common/pnm.c"
    "${APRILTAG_VENDOR_DIR}/common/string_util.c"
    "${APRILTAG_VENDOR_DIR}/common/svd22.c"
    "${APRILTAG_VENDOR_DIR}/common/time_util.c"
    "${APRILTAG_VENDOR_DIR}/common/unionfind.c"
    "${APRILTAG_VENDOR_DIR}/common/workerpool.c"
    "${APRILTAG_VENDOR_DIR}/common/zarray.c"
    "${APRILTAG_VENDOR_DIR}/common/zhash.c"
    "${APRILTAG_VENDOR_DIR}/common/zmaxheap.c"
    "${APRILTAG_VENDOR_DIR}/tag16h5.c"
    "${APRILTAG_VENDOR_DIR}/tag25h9.c"
    "${APRILTAG_VENDOR_DIR}/tag36h11.c"
    "${APRILTAG_VENDOR_DIR}/tagCircle21h7.c"
    "${APRILTAG_VENDOR_DIR}/tagCircle49h12.c"
    "${APRILTAG_VENDOR_DIR}/tagCustom48h12.c"
    "${APRILTAG_VENDOR_DIR}/tagStandard41h12.c"
    "${APRILTAG_VENDOR_DIR}/tagStandard52h13.c")

add_library(apriltag_vendor STATIC ${APRILTAG_VENDOR_SOURCES})
set_target_properties(apriltag_vendor PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(apriltag_vendor PUBLIC "${APRILTAG_VENDOR_DIR}")
target_link_libraries(apriltag_vendor PUBLIC Threads::Threads m)

set(ORIN_UPSTREAM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/upstream")
set(ORIN_CUDA_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/staged/frc971/orin")
file(REMOVE_RECURSE "${ORIN_CUDA_SOURCE_DIR}")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/staged/frc971")
file(COPY "${ORIN_UPSTREAM_ROOT}/frc971/orin"
    DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/staged/frc971")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/overrides/frc971/orin/cuda.h"
    "${ORIN_CUDA_SOURCE_DIR}/cuda.h"
    COPYONLY)

set(ORIN_LINE_FIT_SOURCE "${ORIN_CUDA_SOURCE_DIR}/line_fit_filter.cu")
file(READ "${ORIN_LINE_FIT_SOURCE}" ORIN_LINE_FIT_CONTENT)
string(REPLACE "\r\n" "\n" ORIN_LINE_FIT_CONTENT "${ORIN_LINE_FIT_CONTENT}")

function(replace_staged_function_once content_variable anchor replacement name)
    string(LENGTH "${anchor}" anchor_length)
    string(LENGTH "${${content_variable}}" content_length)
    string(REPLACE "${anchor}" "" content_without_anchor
        "${${content_variable}}")
    string(LENGTH "${content_without_anchor}" content_without_anchor_length)
    math(EXPR match_count
        "(${content_length} - ${content_without_anchor_length}) / ${anchor_length}")
    if(NOT match_count EQUAL 1)
        message(FATAL_ERROR
            "Expected exactly one full-function ${name} anchor in staged "
            "line_fit_filter.cu; found ${match_count}.")
    endif()
    string(REPLACE "${anchor}" "${replacement}" replaced_content
        "${${content_variable}}")
    set(${content_variable} "${replaced_content}" PARENT_SCOPE)
endfunction()

set(ORIN_FIT_LINES_ANCHOR [=[void FitLines(
    const LineFitPoint *line_fit_points_device, size_t points,
    const cub::KeyValuePair<long, MinMaxExtents> *selected_extents_device,
    size_t num_extents, double *errs_device, double *filtered_errs_device,
    Peak *peaks_device, CudaStream *stream) {
  constexpr size_t kThreads = 128;
  const size_t kBlocks = (points + kPointsPerBlock - 2 * kErrorsBuffer - 1) /
                         (kPointsPerBlock - 2 * kErrorsBuffer);

  /*VLOG(1) << "Spawning with " << kThreads << " threads, and " << kBlocks
          << " blocks for " << num_extents << " blob_ids and " << points
          << " points";*/
  DoFitLines<kThreads><<<kBlocks, kThreads, 0, stream->get()>>>(
      line_fit_points_device, points, selected_extents_device, errs_device,
      filtered_errs_device, peaks_device);
}]=])
set(ORIN_FIT_LINES_REPLACEMENT [=[void FitLines(
    const LineFitPoint *line_fit_points_device, size_t points,
    const cub::KeyValuePair<long, MinMaxExtents> *selected_extents_device,
    size_t num_extents, double *errs_device, double *filtered_errs_device,
    Peak *peaks_device, CudaStream *stream) {
  if (points == 0) return;
  constexpr size_t kThreads = 128;
  const size_t kBlocks = (points + kPointsPerBlock - 2 * kErrorsBuffer - 1) /
                         (kPointsPerBlock - 2 * kErrorsBuffer);

  /*VLOG(1) << "Spawning with " << kThreads << " threads, and " << kBlocks
          << " blocks for " << num_extents << " blob_ids and " << points
          << " points";*/
  DoFitLines<kThreads><<<kBlocks, kThreads, 0, stream->get()>>>(
      line_fit_points_device, points, selected_extents_device, errs_device,
      filtered_errs_device, peaks_device);
}]=])

set(ORIN_FIT_QUADS_ANCHOR [=[void FitQuads(
    const Peak *peaks_device, size_t /*peaks*/, const PeakExtents *peak_extents,
    size_t num_extents, const LineFitPoint *line_fit_points_device, int nmaxima,
    const cub::KeyValuePair<long, MinMaxExtents> *selected_extents_device,
    float max_line_fit_mse, double cos_critical_rad, FitQuad *fit_quads_device,
    CudaStream *stream) {
  constexpr size_t kThreads = MaxRankedIndex();
  const size_t kBlocks = num_extents;
  /*VLOG(1) << "Spawning with " << kThreads << " threads, and " << kBlocks
          << " blocks for " << num_extents << " blob_ids";*/
  CHECK_EQ(nmaxima, kNMaxima)
      //<< ": Kernel is compiled and optimized for a fixed nmaxima, please "
      //   "recompile if you want to change it.";
  DoFitQuads<<<kBlocks, kThreads, 0, stream->get()>>>(
      peaks_device, peak_extents, line_fit_points_device,
      selected_extents_device, max_line_fit_mse, cos_critical_rad,
      fit_quads_device);
}]=])
set(ORIN_FIT_QUADS_REPLACEMENT [=[void FitQuads(
    const Peak *peaks_device, size_t /*peaks*/, const PeakExtents *peak_extents,
    size_t num_extents, const LineFitPoint *line_fit_points_device, int nmaxima,
    const cub::KeyValuePair<long, MinMaxExtents> *selected_extents_device,
    float max_line_fit_mse, double cos_critical_rad, FitQuad *fit_quads_device,
    CudaStream *stream) {
  if (num_extents == 0) return;
  constexpr size_t kThreads = MaxRankedIndex();
  const size_t kBlocks = num_extents;
  /*VLOG(1) << "Spawning with " << kThreads << " threads, and " << kBlocks
          << " blocks for " << num_extents << " blob_ids";*/
  CHECK_EQ(nmaxima, kNMaxima);
      //<< ": Kernel is compiled and optimized for a fixed nmaxima, please "
      //   "recompile if you want to change it.";
  DoFitQuads<<<kBlocks, kThreads, 0, stream->get()>>>(
      peaks_device, peak_extents, line_fit_points_device,
      selected_extents_device, max_line_fit_mse, cos_critical_rad,
      fit_quads_device);
}]=])

replace_staged_function_once(ORIN_LINE_FIT_CONTENT
    "${ORIN_FIT_LINES_ANCHOR}" "${ORIN_FIT_LINES_REPLACEMENT}" "FitLines")
replace_staged_function_once(ORIN_LINE_FIT_CONTENT
    "${ORIN_FIT_QUADS_ANCHOR}" "${ORIN_FIT_QUADS_REPLACEMENT}" "FitQuads")
file(WRITE "${ORIN_LINE_FIT_SOURCE}" "${ORIN_LINE_FIT_CONTENT}")

set(ORIN_CUDA_COMPAT_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/compat")
file(MAKE_DIRECTORY "${ORIN_CUDA_COMPAT_INCLUDE_DIR}/cub/iterator")
file(WRITE
    "${ORIN_CUDA_COMPAT_INCLUDE_DIR}/cub/iterator/discard_output_iterator.cuh"
    [=[#pragma once
#include <thrust/iterator/discard_iterator.h>
namespace cub {
template <typename OffsetT>
using DiscardOutputIterator = thrust::discard_iterator<>;
}  // namespace cub
]=])
file(WRITE
    "${ORIN_CUDA_COMPAT_INCLUDE_DIR}/cub/iterator/transform_input_iterator.cuh"
    [=[#pragma once
#include <thrust/iterator/transform_iterator.h>
namespace cub {
template <typename Value, typename UnaryOperation, typename InputIterator>
using TransformInputIterator =
    thrust::transform_iterator<UnaryOperation, InputIterator, Value, Value>;
}  // namespace cub
]=])

set(ORIN_CUDA_SOURCES
    ${ORIN_CUDA_SOURCE_DIR}/971apriltag.cu
    ${ORIN_CUDA_SOURCE_DIR}/apriltag_detect.cu
    ${ORIN_CUDA_SOURCE_DIR}/labeling_allegretti_2019_BKE.cu
    ${ORIN_CUDA_SOURCE_DIR}/line_fit_filter.cu
    ${ORIN_CUDA_SOURCE_DIR}/points.cu
    ${ORIN_CUDA_SOURCE_DIR}/threshold.cu
    ${ORIN_CUDA_SOURCE_DIR}/cuda.cc
    jni/CudaAprilTagDetector.cu)

add_library(orin_apriltag_backend STATIC ${ORIN_CUDA_SOURCES})
target_include_directories(orin_apriltag_backend
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/jni"
    PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}/staged"
        "${ORIN_CUDA_COMPAT_INCLUDE_DIR}"
        "${ORIN_UPSTREAM_ROOT}"
        "${CUDA_CCCL_INCLUDE_DIR}")
target_compile_options(orin_apriltag_backend PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_link_libraries(orin_apriltag_backend
    PUBLIC CUDA::cudart apriltag_vendor Threads::Threads)

add_library(lib971apriltag SHARED
    jni/GpuDetectorJNI.cpp
    jni/DetectorRegistry.cpp)
set_target_properties(lib971apriltag PROPERTIES
    OUTPUT_NAME 971apriltag
    POSITION_INDEPENDENT_CODE ON)
target_include_directories(lib971apriltag PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/jni"
    "${PV_GENERATED_INCLUDE_DIR}")
target_link_libraries(lib971apriltag PRIVATE
    orin_apriltag_backend
    CUDA::cudart
    apriltag_vendor
    JNI::JNI
    Threads::Threads)

add_executable(detector_registry_test
    jni/DetectorRegistry.cpp
    tests/DetectorRegistryTest.cpp)
target_include_directories(detector_registry_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/jni")
target_link_libraries(detector_registry_test PRIVATE Threads::Threads)
add_test(NAME detector_registry_test COMMAND detector_registry_test)
set_tests_properties(detector_registry_test PROPERTIES TIMEOUT 10)

add_executable(cuda_detector_smoke_test
    tests/CudaDetectorSmokeTest.cpp)
target_link_libraries(cuda_detector_smoke_test PRIVATE orin_apriltag_backend)
add_test(NAME cuda_detector_smoke_test COMMAND cuda_detector_smoke_test)
set_tests_properties(cuda_detector_smoke_test PROPERTIES TIMEOUT 180)
