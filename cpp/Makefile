CXX		:= $(shell which g++)
ROOTCONFIG	:= $(shell which root-config)

CXXSTD		:= -std=c++17
OPTS		:= -g

CUINCLUDES	:= -I${CUDA_ROOT}/include -I${CUDNN_ROOT}/include -I${TRT_ROOT}/include -Iinclude
CUFLAGS		:= -L${CUDA_ROOT}/lib64 -lcudart -L${CUDNN_ROOT}/lib64 -lcudnn -L${TRT_ROOT}/lib -lnvinfer -lnvonnxparser
ROOTFLAGS	:= $(shell $(ROOTCONFIG) --libs)

CXXFLAGS	:= $(CUINCLUDES) $(CXXSTD) $(OPTS)
LDFLAGS		:= $(CUFLAGS) $(ROOTFLAGS)

SRCS		:= $(wildcard src/*.cu)

.PHONY: all clean

all:	main

main:	$(SRCS:%.cu=%.o)
	$(CXX) -o main $< $(LDFLAGS)

$(SRCS:%.cu=%.o):%.o: %.cu
	$(CXX) -o $@ -c $(CXXFLAGS) $<

clean: 
	rm -f main *.o
