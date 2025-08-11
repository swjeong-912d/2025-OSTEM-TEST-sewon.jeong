cmake_minimum_required(VERSION 3.5)

set(SOLUTION_TITLE coding_test)

project(
	${SOLUTION_TITLE}_sln
	LANGUAGES CXX
)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(stereo_calibration)
add_subdirectory(tooth_segmentation)