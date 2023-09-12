clean:
	rm -r build

build:
	cmake -S . -B build
	cmake --build build 

clean_build : clean build

run : 
	./build/driver/FingerprintParallel*
	
	

