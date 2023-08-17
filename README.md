# cuLib
Problematic cuda behavior test

# Compile:

```
git clone --recurse-submodules https://github.com/matyro/cuLib.git
cd cuLib

mkdir build
cd build
cmake ..
make
#make test

./tests/cuda/check_cudaLib

python3 python_test.py

```

# Output:
```
#CPP
Randomness seeded to: 1947888300
Sync before kernel
Running kernel
tex_coords: 0.500000
tex_coords: 1.500000
tex_coords: 2.500000
tex_coords: 3.500000
tex_coords: 4.500000
tex_coords: 5.500000
tex_coords: 6.500000
tex_coords: 7.500000
tex_coords: 8.500000
tex_coords: 9.500000
tex_coords: 10.500000
tex_coords: 11.500000
tex_coords: 12.500000
tex_coords: 13.500000
tex_coords: 14.500000
tex_coords: 15.500000
Sync after failed with an illegal memory access was encountered
error copying data to host: an illegal memory access was encountered

# Python
#########################################
Function Style
#########################################
Kernel failed: an illegal memory access was encountered
Traceback (most recent call last):
  File "/cephfs/users/baack/dis/cuLib/build/python_test.py", line 30, in <module>
    main()
  File "/cephfs/users/baack/dis/cuLib/build/python_test.py", line 16, in main
    ret = lerp(query, table.tolist(), 1.0)
RuntimeError: Kernel failed an illegal memory access was encountered 700
```

# "Bugfix"

In `cuda/export/lerp_bindings.cu` switch the `#define` commands at the top of the file with the uncommented ones.
This change from several individual functions calls for the texture to a direct implementation that was copied in.

Resulting in a working example ...


# Other stuff
Building the library `cuda/src/CMakeLists.txt` with `CUDA_SEPARABLE_COMPILATION ON` fails to compile
```
# /usr/local/cuda/bin/crt/link.stub:87:30: error:  redefinition of ‘const unsigned char def_module_id_str
```
