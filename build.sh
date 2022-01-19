echo "Build & run hellosimd..."

echo "deleting previous build..."
rm -r -f build

echo "Creating build folder..."
mkdir build

echo "Compiling..."
clang -Weverything -Wno-padded -Wno-gnu-empty-initializer -Wno-poison-system-directories -std="c99" -oi -o2 -o build/hellosimd src/hellosimd.c

echo "Running..."
(cd build && time ./hellosimd)

