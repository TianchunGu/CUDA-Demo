# 最低版本要求
cmake_minimum_required(VERSION 3.20)
# 项目信息
project(main_program LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 14) # 用于指定CUDA编译器应该使用的CUDA C++标准的版本
set(CMAKE_CUDA_STANDARD_REQUIRED ON)    # 表明如果找不到指定版本的CUDA编译器，将发出错误
set(CMAKE_CXX_STANDARD 14)  # 用于指定 C++ 编译器应该使用的 C++ 标准版本
set(CMAKE_CXX_STANDARD_REQUIRED ON) # 表明如果找不到指定版本的 C++ 编译器，将发出错误

# set(CMAKE_CUDA_FLAGS "-G -g -O0")
# set(CMAKE_CUDA_FLAGS_DEBUG="-G -g -O0")
find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
    message(STATUS "CUDA_LIBRARY_DIRS: ${CUDA_LIBRARY_DIRS}")
else()
message(FATAL_ERROR "Cannot find CUDA")
endif()

# 添加子目录
add_subdirectory(util)
add_subdirectory(program_main)

# 添加新的可执行文件目标
add_executable(main_app app/main.cpp)

# 将可执行文件链接到我们的 CUDA 任务库
# 由于 cuda_task 链接了 util, 链接属性会自动传递过来
target_link_libraries(main_app PRIVATE cuda_task)