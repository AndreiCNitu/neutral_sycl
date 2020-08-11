# Intel SYCL
intel :
	clang++ --gcc-toolchain=/nfs/software/x86_64/gcc/7.4.0 -g -w -fsycl \
	main.cpp params.cpp profiler.cpp comms.cpp shared.cpp data.cpp mesh.cpp shared_data.cpp halos.cpp neutral.cpp neutral_data.cpp \
	-o neutral.sycl
