cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(map)

add_compile_options(-fPIC)
set(CMAKE_CXX_STANDARD 17)
set(CXX_FLAGS "-std=c++17 -lstdc++fs")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -g -pthread")
set(CMAKE_BUILD_TYPE "Debug")
set(PACKAGE_VERSION 1.0)


find_package(OpenMP REQUIRED)
find_package(PCL 1.12 REQUIRED QUIET)
find_package(OpenCV REQUIRED QUIET)
find_package(Eigen3)
find_package(Ceres REQUIRED)


# include directories
include_directories(
	include
	${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

###########
## Build ##
###########

# Range Image Projection

aux_source_directory(./src/common DIR_COM)
aux_source_directory(./src/innolog DIR_LOG)
aux_source_directory(./src/ground DIR_GRD)
aux_source_directory(./src/removert DIR_RMV)

add_executable( ${PROJECT_NAME} ./src/main.cpp ${DIR_COM} ${DIR_LOG} ${DIR_GRD} ${DIR_RMV})
target_compile_options(${PROJECT_NAME} PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(${PROJECT_NAME}
  ${PCL_LIBRARIES} 
  ${OpenCV_LIBRARIES}
  ${OpenMP_CXX_FLAGS}
  stdc++fs
)


