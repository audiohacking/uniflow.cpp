cmake_minimum_required(VERSION 3.16)
project(uniflow-audio LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(UNIFLOW_METAL "Build with ggml-metal backend" ON)
option(UNIFLOW_CUDA  "Build with ggml-cuda backend"  OFF)
option(UNIFLOW_BUILD_TESTS "Build C++ regression tests" ON)

if (APPLE AND UNIFLOW_METAL)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()
if (UNIFLOW_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
endif()

add_subdirectory(third_party/ggml)

find_package(PkgConfig REQUIRED)
pkg_check_modules(SENTENCEPIECE REQUIRED sentencepiece)
# Homebrew sentencepiece headers pull in absl::Status; link absl explicitly.
pkg_check_modules(ABSL_STATUS REQUIRED absl_status)

add_library(uniflow_audio_core
    src/backend.cpp
    src/t5_encoder.cpp
    src/tokenizer.cpp
    src/scheduler.cpp
    src/wav_io.cpp
)
target_include_directories(uniflow_audio_core PUBLIC
    src
    third_party/pocketfft
    ${SENTENCEPIECE_INCLUDE_DIRS}
    ${ABSL_STATUS_INCLUDE_DIRS}
)
target_link_libraries(uniflow_audio_core PUBLIC
    ggml
    ${SENTENCEPIECE_LIBRARIES}
    ${ABSL_STATUS_LIBRARIES}
)
target_link_directories(uniflow_audio_core PUBLIC
    ${SENTENCEPIECE_LIBRARY_DIRS}
    ${ABSL_STATUS_LIBRARY_DIRS}
)

add_executable(uniflow-audio src/main.cpp)
target_link_libraries(uniflow-audio PRIVATE uniflow_audio_core)

if (UNIFLOW_BUILD_TESTS)
    enable_testing()
    add_executable(uniflow_test_smoke tests/cpp/test_smoke.cpp)
    target_link_libraries(uniflow_test_smoke PRIVATE uniflow_audio_core)
    add_test(NAME smoke COMMAND uniflow_test_smoke)

    add_executable(uniflow_test_scheduler tests/cpp/test_scheduler.cpp)
    target_link_libraries(uniflow_test_scheduler PRIVATE uniflow_audio_core)
    add_test(NAME scheduler COMMAND uniflow_test_scheduler)

    add_executable(uniflow_test_wav_io tests/cpp/test_wav_io.cpp)
    target_link_libraries(uniflow_test_wav_io PRIVATE uniflow_audio_core)
    add_test(NAME wav_io COMMAND uniflow_test_wav_io)
endif()
