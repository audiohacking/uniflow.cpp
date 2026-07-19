# Makefile for uniflow-audio
# Wraps CMake commands for convenience

BUILD_DIR_CPU   := build-cpu
BUILD_DIR_METAL := build-metal
BUILD_DIR_CUDA  := build-cuda

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
	rm -rf $(BUILD_DIR_CPU) $(BUILD_DIR_METAL) $(BUILD_DIR_CUDA)

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

# Convert only T5 for now (safe). DiT/VAE converters are intentionally absent
# until tensor maps are verified against UniFlow HF weights — see DEVELOPMENT.md.
# Convert UniFlow-Audio-v1.1-Small (T2A subset). Requires models/uniflow-small/.
.PHONY: convert-small
convert-small:
	@test -f models/uniflow-small/model.safetensors || \
		(echo "Missing models/uniflow-small — see DEVELOPMENT.md"; exit 1)
	@mkdir -p models
	@echo "=== DiT (F16) ==="
	python3 convert/convert_dit.py models/uniflow-small -o models/dit.gguf --dtype f16 --variant small
	@echo "=== VAE (F16, weight_norm fused) ==="
	python3 convert/convert_vae.py models/uniflow-small -o models/vae.gguf --dtype f16
	@echo "=== Instructions ==="
	python3 convert/convert_instructions.py models/uniflow-small -o models/instructions.gguf
	@echo "=== T5 (optional if missing) ==="
	@if [ ! -f models/t5_encoder.gguf ]; then \
		python3 convert/convert_t5_encoder.py google/flan-t5-large -o models/t5_encoder.gguf --dtype f32; \
		cp ~/.cache/huggingface/hub/models--google--flan-t5-large/snapshots/*/spiece.model models/ 2>/dev/null || true; \
	fi
	@ls -lh models/*.gguf models/spiece.model 2>/dev/null || ls -lh models/*.gguf

.PHONY: convert-t5
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
	@echo "  make convert-t5       Convert Flan-T5 encoder only"
	@echo "  make convert-small|base|large   Convert variant pack → dist/hf/"
	@echo "  make download-gguf    Fetch Small GGUF pack from HF (default)"
	@echo "  make download-gguf-small|base|large"
	@echo "  make clean            Remove build dirs"

# Variant converts → dist/hf/uniflow-audio-v1.1-{variant}/
.PHONY: convert-base convert-large convert-xlarge
convert-base:
	./scripts/convert_variant.sh base
convert-large:
	./scripts/convert_variant.sh large
convert-xlarge:
	./scripts/convert_variant.sh xlarge

.PHONY: download-gguf download-gguf-small download-gguf-base download-gguf-large
download-gguf: download-gguf-small
download-gguf-small:
	./scripts/download_gguf.sh small
download-gguf-base:
	./scripts/download_gguf.sh base
download-gguf-large:
	./scripts/download_gguf.sh large
