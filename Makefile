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
	rm -f neutral.sycl
