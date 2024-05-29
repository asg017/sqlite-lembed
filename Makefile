
COMMIT=$(shell git rev-parse HEAD)
VERSION=$(shell cat VERSION)
DATE=$(shell date +'%FT%TZ%z')

ifndef CC
CC=gcc
endif
ifndef AR
AR=ar
endif

ifeq ($(shell uname -s),Darwin)
CONFIG_DARWIN=y
else ifeq ($(OS),Windows_NT)
CONFIG_WINDOWS=y
else
CONFIG_LINUX=y
endif

ifdef CONFIG_DARWIN
LOADABLE_EXTENSION=dylib
CFLAGS+=-framework Accelerate -framework Foundation -framework Metal -framework MetalKit
LLAMA_CMAKE_FLAGS+=-DLLAMA_METAL_EMBED_LIBRARY=1
endif

ifdef CONFIG_LINUX
LOADABLE_EXTENSION=so
LLAMA_CMAKE_FLAGS+=-DCMAKE_POSITION_INDEPENDENT_CODE=ON
endif

ifdef CONFIG_WINDOWS
LOADABLE_EXTENSION=dll
endif


ifdef python
PYTHON=$(python)
else
PYTHON=python3
endif

ifdef release
LLAMA_CMAKE_FLAGS+=-DCMAKE_BUILD_TYPE=Release
else
LLAMA_CMAKE_FLAGS+=-DCMAKE_BUILD_TYPE=Debug
endif

prefix=dist

$(prefix):
	mkdir -p $(prefix)

TARGET_LOADABLE=$(prefix)/lembed0.$(LOADABLE_EXTENSION)
TARGET_STATIC=$(prefix)/libsqlite_lembed0.a
TARGET_STATIC_H=$(prefix)/sqlite-lembed.h

loadable: $(TARGET_LOADABLE)
static: $(TARGET_STATIC)


LLAMA_BUILD_DIR=$(prefix)/.llama

ifdef CONFIG_WINDOWS
LLAMA_BUILD_TARGETS=$(LLAMA_BUILD_DIR)/Debug/llama.lib $(LLAMA_BUILD_DIR)/Debug/ggml_static.lib
else
LLAMA_BUILD_TARGETS=$(LLAMA_BUILD_DIR)/libllama.a $(LLAMA_BUILD_DIR)/libggml_static.a
endif


$(LLAMA_BUILD_DIR):
	rm -rf @ || true
	cmake \
		-S vendor/llama.cpp \
		-B $@ \
		-DLLAMA_STATIC=1 $(LLAMA_CMAKE_FLAGS)


$(LLAMA_BUILD_TARGETS): $(LLAMA_BUILD_DIR)
	cmake --build $(LLAMA_BUILD_DIR) -t ggml_static -t llama


$(TARGET_LOADABLE): sqlite-lembed.c sqlite-lembed.h $(LLAMA_BUILD_TARGETS) $(prefix)
	gcc \
		-fPIC -shared \
		-Ivendor/sqlite \
		-Ivendor/llama.cpp \
		-O3 \
		$(CFLAGS) \
		-lstdc++ \
		$< $(LLAMA_BUILD_TARGETS) \
		-o $@



sqlite-lembed.h: sqlite-lembed.h.tmpl VERSION
	VERSION=$(shell cat VERSION) \
	DATE=$(shell date -r VERSION +'%FT%TZ%z') \
	SOURCE=$(shell git log -n 1 --pretty=format:%H -- VERSION) \
	envsubst < $< > $@

test-loadable:
	echo 4


FORMAT_FILES=sqlite-lembed.c
format: $(FORMAT_FILES)
	clang-format -i $(FORMAT_FILES)
