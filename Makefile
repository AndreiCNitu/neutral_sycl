COMPUTECPP_FLAGS = $(shell $(COMPUTECPP_PACKAGE_ROOT_DIR)/bin/computecpp_info --dump-device-compiler-flags)
FLAGS= -O2
EXTRA_FLAGS=

# ComputeCpp
computecpp : main.o params.o profiler.o comms.o shared.o data.o mesh.o shared_data.o halos.o neutral.o neutral_data.o main.sycl params.sycl profiler.sycl comms.sycl shared.sycl data.sycl mesh.sycl shared_data.sycl halos.sycl neutral.sycl neutral_data.sycl
	$(CXX) $(FLAGS) -std=c++11 -DSYCL \
	main.o params.o profiler.o comms.o shared.o data.o mesh.o shared_data.o halos.o neutral.o neutral_data.o \
	$(EXTRA_FLAGS) -L$(COMPUTECPP_PACKAGE_ROOT_DIR)/lib -lComputeCpp -lOpenCL -Wl,--rpath=$(COMPUTECPP_PACKAGE_ROOT_DIR)/lib/ -o neutral

main.o : main.cpp main.sycl
	$(CXX) $(FLAGS) -std=c++11 -DSYCL -DCL_TARGET_OPENCL_VERSION=220 \
	main.cpp -c \
	-I$(COMPUTECPP_PACKAGE_ROOT_DIR)/include -include main.sycl $(EXTRA_FLAGS) -o $@

params.o : params.cpp params.sycl
	$(CXX) $(FLAGS) -std=c++11 -DSYCL -DCL_TARGET_OPENCL_VERSION=220 \
	params.cpp -c \
	-I$(COMPUTECPP_PACKAGE_ROOT_DIR)/include -include params.sycl $(EXTRA_FLAGS) -o $@

profiler.o : profiler.cpp profiler.sycl
	$(CXX) $(FLAGS) -std=c++11 -DSYCL -DCL_TARGET_OPENCL_VERSION=220 \
	profiler.cpp -c \
	-I$(COMPUTECPP_PACKAGE_ROOT_DIR)/include -include profiler.sycl $(EXTRA_FLAGS) -o $@

comms.o : comms.cpp comms.sycl
	$(CXX) $(FLAGS) -std=c++11 -DSYCL -DCL_TARGET_OPENCL_VERSION=220 \
	comms.cpp -c \
	-I$(COMPUTECPP_PACKAGE_ROOT_DIR)/include -include comms.sycl $(EXTRA_FLAGS) -o $@

shared.o : shared.cpp shared.sycl
	$(CXX) $(FLAGS) -std=c++11 -DSYCL -DCL_TARGET_OPENCL_VERSION=220 \
	shared.cpp -c \
	-I$(COMPUTECPP_PACKAGE_ROOT_DIR)/include -include shared.sycl $(EXTRA_FLAGS) -o $@

data.o : data.cpp data.sycl
	$(CXX) $(FLAGS) -std=c++11 -DSYCL -DCL_TARGET_OPENCL_VERSION=220 \
	data.cpp -c \
	-I$(COMPUTECPP_PACKAGE_ROOT_DIR)/include -include data.sycl $(EXTRA_FLAGS) -o $@

mesh.o : mesh.cpp mesh.sycl
	$(CXX) $(FLAGS) -std=c++11 -DSYCL -DCL_TARGET_OPENCL_VERSION=220 \
	mesh.cpp -c \
	-I$(COMPUTECPP_PACKAGE_ROOT_DIR)/include -include mesh.sycl $(EXTRA_FLAGS) -o $@

shared_data.o : shared_data.cpp shared_data.sycl
	$(CXX) $(FLAGS) -std=c++11 -DSYCL -DCL_TARGET_OPENCL_VERSION=220 \
	shared_data.cpp -c \
	-I$(COMPUTECPP_PACKAGE_ROOT_DIR)/include -include shared_data.sycl $(EXTRA_FLAGS) -o $@

halos.o : halos.cpp halos.sycl
	$(CXX) $(FLAGS) -std=c++11 -DSYCL -DCL_TARGET_OPENCL_VERSION=220 \
	halos.cpp -c \
	-I$(COMPUTECPP_PACKAGE_ROOT_DIR)/include -include halos.sycl $(EXTRA_FLAGS) -o $@

neutral.o : neutral.cpp neutral.sycl
	$(CXX) $(FLAGS) -std=c++11 -DSYCL -DCL_TARGET_OPENCL_VERSION=220 \
	neutral.cpp -c \
	-I$(COMPUTECPP_PACKAGE_ROOT_DIR)/include -include neutral.sycl $(EXTRA_FLAGS) -o $@

neutral_data.o : neutral_data.cpp neutral_data.sycl
	$(CXX) $(FLAGS) -std=c++11 -DSYCL -DCL_TARGET_OPENCL_VERSION=220 \
	neutral_data.cpp -c \
	-I$(COMPUTECPP_PACKAGE_ROOT_DIR)/include -include neutral_data.sycl $(EXTRA_FLAGS) -o $@

main.sycl : main.cpp
	$(COMPUTECPP_PACKAGE_ROOT_DIR)/bin/compute++ $(FLAGS) -no-serial-memop -DSYCL -DCL_TARGET_OPENCL_VERSION=220 \
	main.cpp \
	$(COMPUTECPP_FLAGS) -c -I$(COMPUTECPP_PACKAGE_ROOT_DIR)/include -o $@

params.sycl : params.cpp
	$(COMPUTECPP_PACKAGE_ROOT_DIR)/bin/compute++ $(FLAGS) -no-serial-memop -DSYCL -DCL_TARGET_OPENCL_VERSION=220 \
	params.cpp \
	$(COMPUTECPP_FLAGS) -c -I$(COMPUTECPP_PACKAGE_ROOT_DIR)/include -o $@

profiler.sycl : profiler.cpp
	$(COMPUTECPP_PACKAGE_ROOT_DIR)/bin/compute++ $(FLAGS) -no-serial-memop -DSYCL -DCL_TARGET_OPENCL_VERSION=220 \
	profiler.cpp \
	$(COMPUTECPP_FLAGS) -c -I$(COMPUTECPP_PACKAGE_ROOT_DIR)/include -o $@

comms.sycl : comms.cpp
	$(COMPUTECPP_PACKAGE_ROOT_DIR)/bin/compute++ $(FLAGS) -no-serial-memop -DSYCL -DCL_TARGET_OPENCL_VERSION=220 \
	comms.cpp \
	$(COMPUTECPP_FLAGS) -c -I$(COMPUTECPP_PACKAGE_ROOT_DIR)/include -o $@

shared.sycl : shared.cpp
	$(COMPUTECPP_PACKAGE_ROOT_DIR)/bin/compute++ $(FLAGS) -no-serial-memop -DSYCL -DCL_TARGET_OPENCL_VERSION=220 \
	shared.cpp \
	$(COMPUTECPP_FLAGS) -c -I$(COMPUTECPP_PACKAGE_ROOT_DIR)/include -o $@

data.sycl : data.cpp
	$(COMPUTECPP_PACKAGE_ROOT_DIR)/bin/compute++ $(FLAGS) -no-serial-memop -DSYCL -DCL_TARGET_OPENCL_VERSION=220 \
	data.cpp \
	$(COMPUTECPP_FLAGS) -c -I$(COMPUTECPP_PACKAGE_ROOT_DIR)/include -o $@

mesh.sycl : mesh.cpp
	$(COMPUTECPP_PACKAGE_ROOT_DIR)/bin/compute++ $(FLAGS) -no-serial-memop -DSYCL -DCL_TARGET_OPENCL_VERSION=220 \
	mesh.cpp \
	$(COMPUTECPP_FLAGS) -c -I$(COMPUTECPP_PACKAGE_ROOT_DIR)/include -o $@

shared_data.sycl : shared_data.cpp
	$(COMPUTECPP_PACKAGE_ROOT_DIR)/bin/compute++ $(FLAGS) -no-serial-memop -DSYCL -DCL_TARGET_OPENCL_VERSION=220 \
	shared_data.cpp \
	$(COMPUTECPP_FLAGS) -c -I$(COMPUTECPP_PACKAGE_ROOT_DIR)/include -o $@

halos.sycl : halos.cpp
	$(COMPUTECPP_PACKAGE_ROOT_DIR)/bin/compute++ $(FLAGS) -no-serial-memop -DSYCL -DCL_TARGET_OPENCL_VERSION=220 \
	halos.cpp \
	$(COMPUTECPP_FLAGS) -c -I$(COMPUTECPP_PACKAGE_ROOT_DIR)/include -o $@

neutral.sycl : neutral.cpp
	$(COMPUTECPP_PACKAGE_ROOT_DIR)/bin/compute++ $(FLAGS) -no-serial-memop -DSYCL -DCL_TARGET_OPENCL_VERSION=220 \
	neutral.cpp \
	$(COMPUTECPP_FLAGS) -c -I$(COMPUTECPP_PACKAGE_ROOT_DIR)/include -o $@

neutral_data.sycl : neutral_data.cpp
	$(COMPUTECPP_PACKAGE_ROOT_DIR)/bin/compute++ $(FLAGS) -no-serial-memop -DSYCL -DCL_TARGET_OPENCL_VERSION=220 \
	neutral_data.cpp \
	$(COMPUTECPP_FLAGS) -c -I$(COMPUTECPP_PACKAGE_ROOT_DIR)/include -o $@


# Intel SYCL
llvm :
	clang++ --gcc-toolchain=/nfs/software/x86_64/gcc/7.4.0 -g -w -fsycl -DSYCL \
	main.cpp params.cpp profiler.cpp comms.cpp shared.cpp data.cpp mesh.cpp shared_data.cpp halos.cpp neutral.cpp neutral_data.cpp \
	-o neutral.sycl

# hipSYCL (2080 Ti)
2080ti :
	syclcc -std=c++17 -O3 -w --hipsycl-gpu-arch=sm_75 --hipsycl-platform=cuda \
	main.cpp params.cpp profiler.cpp comms.cpp shared.cpp data.cpp mesh.cpp shared_data.cpp halos.cpp neutral.cpp neutral_data.cpp \
	-o neutral.sycl


.PHONY: all check clean

clean:
	rm -f neutral
	rm -f *.o
	rm -f *.sycl
