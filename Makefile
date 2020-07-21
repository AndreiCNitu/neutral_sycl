# Intel SYCL
intel :
	clang++ -w -fsycl main.cpp params.cpp profiler.cpp comms.cpp shared.cpp data.cpp mesh.cpp shared_data.cpp halos.cpp neutral.cpp neutral_data.cpp -o neutral.sycl
