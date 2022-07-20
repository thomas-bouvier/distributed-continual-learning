MAIN=rehearsal
SOURCES=rehearsal.cpp stream_loader.cpp distributed_stream_loader.cpp
FLAGS=-O3 -Wall -std=c++14 -fPIC -D_GLIBCXX_USE_CXX11_ABI=0

# Bogdan:
#TORCH_ROOT?=$(HOME)/.local/lib/python3.10/site-packages/torch
# Thomas: before:
TORCH_ROOT?=$(HOME)/.conda/envs/horovod-py39/lib/python3.9/site-packages/torch
TORCH_INCLUDE=-I$(TORCH_ROOT)/include -I$(TORCH_ROOT)/include/torch/csrc/api/include

THALLIUM_INCLUDE?=$(shell pkg-config --cflags-only-I thallium)
THALLIUM_LIBS?=$(shell pkg-config --libs thallium)

INCLUDES=$(shell python -m pybind11 --includes) $(TORCH_INCLUDE) $(THALLIUM_INCLUDE)
LIBS=-L$(TORCH_ROOT)/lib -ltorch $(THALLIUM_LIBS) -lc10 -ltorch_cpu
EXT=$(shell python3-config --extension-suffix)
CC=g++

all:
	$(CC) -shared $(FLAGS) $(INCLUDES) $(SOURCES) -o $(MAIN)$(EXT) $(LIBS)
test:
	$(CC) -g $(FLAGS) $(INCLUDES) distributed_stream_loader.cpp main.cpp -o $(MAIN) $(LIBS)
clean:
	rm -rf $(MAIN)*.so *~
