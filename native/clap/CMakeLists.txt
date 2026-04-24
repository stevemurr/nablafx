cmake_minimum_required(VERSION 3.22)

# The concrete plugin name comes from plugin_meta.json, not CMake. We build a
# single generic dylib; ``build.sh`` packages it as a per-model .clap bundle
# with the model-specific metadata.
project(nablafx_clap CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT APPLE)
    message(FATAL_ERROR "v1 targets macOS arm64 only. Cross/universal builds are v2.")
endif()

if(NOT CMAKE_OSX_ARCHITECTURES)
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "" FORCE)
endif()

include(cmake/FetchCLAP.cmake)
include(cmake/FetchOnnxRuntime.cmake)
include(cmake/FetchJson.cmake)

add_library(nablafx_clap MODULE
    src/nablafx_plugin.cpp
    src/meta.cpp
    src/ort_session.cpp
    src/param_id.cpp
)

target_include_directories(nablafx_clap PRIVATE
    ${CLAP_INCLUDE_DIR}
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${JSON_INCLUDE_DIR}
)

target_link_libraries(nablafx_clap PRIVATE
    ${ONNXRUNTIME_LIBRARY}
    "-framework CoreFoundation"
)

# Install the onnxruntime dylib next to ours and make the plugin find it via
# @loader_path/../Frameworks once it's inside the .clap bundle.
set_target_properties(nablafx_clap PROPERTIES
    PREFIX ""
    SUFFIX ".so"                 # build.sh renames to the bundle executable
    INSTALL_RPATH "@loader_path/../Frameworks"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Expose artifacts to build.sh via a tiny config file
configure_file(
    ${CMAKE_SOURCE_DIR}/cmake/build_config.cmake.in
    ${CMAKE_BINARY_DIR}/build_config.sh
    @ONLY
)
