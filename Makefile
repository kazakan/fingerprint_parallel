.PHONY: build test

clean:
	rm -r build

build:
	cmake -S . -B build
	cmake --build build 

cbuild : clean build

run : 
	./build/driver/FingerprintParallel*
	
test : build
	cd ./build/test && ctest --output-on-failure