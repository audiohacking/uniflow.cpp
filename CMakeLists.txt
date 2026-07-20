cmake_minimum_required(VERSION 3.16)
project(uniflow-audio LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(UNIFLOW_METAL "Build with ggml-metal backend" ON)
option(UNIFLOW_CUDA  "Build with ggml-cuda backend"  OFF)
option(UNIFLOW_BUILD_TESTS "Build C++ regression tests" ON)
option(UNIFLOW_WEB "Build Emscripten WebGPU demo (uniflow-web)" OFF)

if (EMSCRIPTEN)
    set(UNIFLOW_METAL OFF CACHE BOOL "" FORCE)
    set(UNIFLOW_CUDA OFF CACHE BOOL "" FORCE)
    set(UNIFLOW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(UNIFLOW_WEB ON CACHE BOOL "" FORCE)
    set(GGML_WEBGPU ON CACHE BOOL "" FORCE)
    set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
endif()

if (APPLE AND UNIFLOW_METAL AND NOT EMSCRIPTEN)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()
if (UNIFLOW_CUDA AND NOT EMSCRIPTEN)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
endif()

add_subdirectory(third_party/ggml)

set(UNIFLOW_CORE_SOURCES
    src/backend.cpp
    src/t5_encoder.cpp
    src/scheduler.cpp
    src/wav_io.cpp
    src/content_adapter.cpp
    src/dit.cpp
    src/vae.cpp
    src/instructions.cpp
    src/pipeline.cpp
)

if (NOT EMSCRIPTEN)
    list(APPEND UNIFLOW_CORE_SOURCES src/tokenizer.cpp)
endif()

add_library(uniflow_audio_core ${UNIFLOW_CORE_SOURCES})
target_include_directories(uniflow_audio_core PUBLIC src third_party/pocketfft)

if (NOT EMSCRIPTEN)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(SENTENCEPIECE REQUIRED sentencepiece)
    pkg_check_modules(ABSL_STATUS REQUIRED absl_status)
    target_include_directories(uniflow_audio_core PUBLIC
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
else()
    target_link_libraries(uniflow_audio_core PUBLIC ggml)
    target_compile_definitions(uniflow_audio_core PUBLIC UNIFLOW_WEB_BUILD=1)
endif()

if (NOT EMSCRIPTEN)
    add_executable(uniflow-audio src/main.cpp)
    target_link_libraries(uniflow-audio PRIVATE uniflow_audio_core)
endif()

if (EMSCRIPTEN AND UNIFLOW_WEB)
    add_executable(uniflow-web src/web_api.cpp)
    target_link_libraries(uniflow-web PRIVATE uniflow_audio_core)
    # Use ASYNCIFY (not JSPI): WebGPU WaitAny needs to suspend, and JSPI
    # requires WebAssembly.promising + embind async() wrappers — fragile on
    # stock Chrome. ASYNCIFY works without browser flags.
    target_link_options(uniflow-web PRIVATE
        "SHELL:-sALLOW_MEMORY_GROWTH=1"
        "SHELL:-sINITIAL_MEMORY=536870912"
        "SHELL:-sMAXIMUM_MEMORY=4294967296"
        "SHELL:-sFORCE_FILESYSTEM=1"
        "SHELL:-sENVIRONMENT=web"
        "SHELL:--bind"
        "SHELL:-sASYNCIFY"
        "SHELL:-sASYNCIFY_STACK_SIZE=1048576"
        "SHELL:-sEXPORTED_RUNTIME_METHODS=['FS','HEAPF32','UTF8ToString']"
    )
    set_target_properties(uniflow-web PROPERTIES
        OUTPUT_NAME "uniflow-web"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/web"
    )
endif()

if (UNIFLOW_BUILD_TESTS AND NOT EMSCRIPTEN)
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

    add_executable(uniflow_test_vae_load tests/cpp/test_vae_load.cpp)
    target_link_libraries(uniflow_test_vae_load PRIVATE uniflow_audio_core)
    add_test(NAME vae_load COMMAND uniflow_test_vae_load WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()
