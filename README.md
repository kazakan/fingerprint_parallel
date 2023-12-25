# Fingerprint Parallel

Parallel algorithm for Fingerprint matching.

## Setting development environment

Following libraries are required.

- OpenCL
- FreeImage
- gtest (for testing)

### ubuntu

```shell
sudo apt-get update
sudo apt-get install build-essential ocl-icd-opencl-dev libfreeimage3 libfreeimage-dev gtest
```

### Windows Msys2

This will be added later.

## Defined tasks

Following tasks are defined using `makefile`.

- `clean` : clean artifacts. This removes all files in `build` directory.
- `build` : build artifacts.
- `cbuild` : clean and build.
- `run` : run driver program.
- `test` : build and run test program.

To run task use command `make <TASK_NAME>` at the root of this repository.
