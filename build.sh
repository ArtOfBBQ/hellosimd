APP_NAME="hellosimd"
CMDLARGS=""

echo "Building $APP_NAME... with arguments: [$CMDLARGS] (this shell script must be run from the app's root directory)"

echo "deleting previous build..."
rm -r -f build

echo "Creating build folder..."
mkdir build

echo "Compiling $APP_NAME..."
clang -Weverything -Wno-padded -Wno-gnu-empty-initializer -Wno-poison-system-directories $MAC_FRAMEWORKS -lstdc++ -std="c99" -o3 -o build/$APP_NAME src/hellosimd.c
# gcc -fsanitize=undefined -g -o3 $MAC_FRAMEWORKS -lstdc++ -std="c99" -o build/$APP_NAME src/hellosimd.c

# echo "Running $APP_NAME"
(cd build && time ./$APP_NAME)

