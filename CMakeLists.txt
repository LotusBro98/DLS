cmake_minimum_required(VERSION 3.9)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(simplenet)

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} ./include/)

set(CMAKE_CXX_STANDARD 11)

add_library(simplenet SHARED
	./src/net/Layer.cpp ./src/net/Net.cpp ./src/net/Dataset.cpp ./src/net/funcs.cpp)

add_library(dls SHARED
	./src/dls.cpp)

add_executable(dls_train
	./src/dls_train.cpp)

target_link_libraries( dls_train simplenet dls pthread ${OpenCV_LIBS} )
