CXX		:= $(shell which g++)
ROOTCONFIG	:= $(shell which root-config)
CUDA_ROOT	:= $(shell dirname ${CUDACXX})/../

CXXSTD		:= -std=c++17
OPTS		:= -g

CUINCLUDES	:= -I${CUDA_ROOT}/include -I${CUDNN_ROOT}/include -I${TRT_ROOT}/include -Iinclude
CUFLAGS		:= -L${CUDA_ROOT}/lib64 -lcudart -L${CUDNN_ROOT}/lib64 -lcudnn -L${TRT_ROOT}/lib -lnvinfer -lnvonnxparser
ROOTFLAGS	:= $(shell $(ROOTCONFIG) --libs)

CXXFLAGS	:= $(CUINCLUDES) $(CXXSTD) $(OPTS)
LDFLAGS		:= $(CUFLAGS) $(ROOTFLAGS)

SRCS		:= $(wildcard *.cpp)

.PHONY: all clean

all:	main

main:	$(SRCS:%.cpp=%.o)
	$(CXX) -o main $< $(LDFLAGS)

$(SRCS:%.cpp=%.o):%.o: %.cpp
	$(CXX) -o $@ -c $(CXXFLAGS) $<

clean: 
	rm -f main *.o
