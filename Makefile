# Makefile for uniflow-audio
# Wraps CMake commands for convenience

BUILD_DIR_CPU   := build-cpu
BUILD_DIR_METAL := build-metal
BUILD_DIR_CUDA  := build-cuda
BUILD_DIR_WEB   := build-web

CMAKE_FLAGS := -DCMAKE_BUILD_TYPE=Release

.PHONY: all
all: metal

.PHONY: cpu
cpu:
	@mkdir -p $(BUILD_DIR_CPU)
	cmake -B $(BUILD_DIR_CPU) -S . $(CMAKE_FLAGS) \
		-DUNIFLOW_METAL=OFF \
		-DUNIFLOW_CUDA=OFF
	cmake --build $(BUILD_DIR_CPU) --parallel
	@echo "CPU build complete: $(BUILD_DIR_CPU)/uniflow-audio"

.PHONY: metal
metal:
	@mkdir -p $(BUILD_DIR_METAL)
	cmake -B $(BUILD_DIR_METAL) -S . $(CMAKE_FLAGS) \
		-DUNIFLOW_METAL=ON \
		-DUNIFLOW_CUDA=OFF
	cmake --build $(BUILD_DIR_METAL) --parallel
	@echo "Metal build complete: $(BUILD_DIR_METAL)/uniflow-audio"

.PHONY: cuda
cuda:
	@mkdir -p $(BUILD_DIR_CUDA)
	cmake -B $(BUILD_DIR_CUDA) -S . $(CMAKE_FLAGS) \
		-DUNIFLOW_METAL=OFF \
		-DUNIFLOW_CUDA=ON
	cmake --build $(BUILD_DIR_CUDA) --parallel
	@echo "CUDA build complete: $(BUILD_DIR_CUDA)/uniflow-audio"

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR_CPU) $(BUILD_DIR_METAL) $(BUILD_DIR_CUDA) $(BUILD_DIR_WEB)

.PHONY: web
web:
	@command -v emcmake >/dev/null || (echo "emcmake not found — install emscripten"; exit 1)
	@./scripts/fetch_emdawnwebgpu.sh
	@mkdir -p $(BUILD_DIR_WEB)
	emcmake cmake -B $(BUILD_DIR_WEB) -S . $(CMAKE_FLAGS) \
		-DUNIFLOW_METAL=OFF \
		-DUNIFLOW_CUDA=OFF \
		-DUNIFLOW_BUILD_TESTS=OFF \
		-DGGML_WEBGPU=ON \
		-DGGML_WEBGPU_JSPI=ON \
		-DGGML_OPENMP=OFF \
		-DEMDAWNWEBGPU_DIR="$(CURDIR)/third_party/emdawnwebgpu_pkg"
	emmake cmake --build $(BUILD_DIR_WEB) --target uniflow-web --parallel
	@echo "Web build complete: web/uniflow-web.{js,wasm}"
	@echo "Serve: npx --yes serve web -p 8080"

.PHONY: rebuild
rebuild: clean all

# Run all regression tests (C++ ctest + Python convert conventions)
.PHONY: test
test: cpu
	cd $(BUILD_DIR_CPU) && ctest --output-on-failure
	python3 tests/python/test_convert_conventions.py

.PHONY: test-metal
test-metal: metal
	cd $(BUILD_DIR_METAL) && ctest --output-on-failure
	python3 tests/python/test_convert_conventions.py

.PHONY: test-smoke
test-smoke: cpu
	$(BUILD_DIR_CPU)/uniflow-audio --smoke-test

.PHONY: test-smoke-metal
test-smoke-metal: metal
	$(BUILD_DIR_METAL)/uniflow-audio --smoke-test

# Convert UniFlow packs → dist/hf/uniflow-audio-v1.1-{variant}/
# QUANT=F16|Q8_0|Q4_0|all (default F16)
.PHONY: convert-small convert-base convert-large convert-xlarge convert-t5
convert-small:
	./scripts/convert_variant.sh small $(or $(QUANT),F16)
convert-base:
	./scripts/convert_variant.sh base $(or $(QUANT),F16)
convert-large:
	./scripts/convert_variant.sh large $(or $(QUANT),F16)
convert-xlarge:
	./scripts/convert_variant.sh xlarge $(or $(QUANT),F16)

convert-t5:
	@mkdir -p models
	@echo "Converting Flan-T5-large encoder → models/t5_encoder.gguf"
	python3 convert/convert_t5_encoder.py google/flan-t5-large -o models/t5_encoder.gguf --dtype f32
	@cp ~/.cache/huggingface/hub/models--google--flan-t5-large/snapshots/*/spiece.model models/ 2>/dev/null || \
		echo "NOTE: download google/flan-t5-large to obtain spiece.model"

.PHONY: help
help:
	@echo "uniflow.cpp targets:"
	@echo "  make metal|cpu|cuda   Build inference binary"
	@echo "  make test             Run C++ + Python regression tests"
	@echo "  make test-smoke       CLI smoke test (CPU)"
	@echo "  make convert-small|base|large   Pack → dist/hf/ (QUANT=F16|Q8_0|Q4_0|all)"
	@echo "  make web              Emscripten WebGPU demo → web/"
	@echo "  make download-gguf            Base Q8_0 (QUANT=… to override)"
	@echo "  make download-gguf-small|base|large"
	@echo "  make clean                    Remove build dirs"

.PHONY: download-gguf download-gguf-small download-gguf-base download-gguf-large
# QUANT=F16|Q8_0|Q4_0|all — default download is base Q8_0
download-gguf: download-gguf-base
download-gguf-small:
	./scripts/download_gguf.sh small $(or $(QUANT),Q8_0)
download-gguf-base:
	./scripts/download_gguf.sh base $(or $(QUANT),Q8_0)
download-gguf-large:
	./scripts/download_gguf.sh large $(or $(QUANT),Q8_0)
