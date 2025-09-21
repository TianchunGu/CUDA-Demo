# 最低版本要求
cmake_minimum_required(VERSION 3.20)
# 项目信息
project(main_app LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 14) 
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 寻找 CUDA 包。这会创建一个导入目标 `CUDA::toolkit`
find_package(CUDAToolkit REQUIRED)
# message(STATUS "Found CUDA Version: ${CUDA_VERSION_STRING}") # (可选) 打印版本信息

# 添加子目录
add_subdirectory(util)
add_subdirectory(cuda_task)

# 添加新的可执行文件目标
add_executable(main_app app/main.cpp)

# 将可执行文件链接到我们的 CUDA 任务库
# PRIVATE 表示 main_app 依赖 cuda_task，但这个依赖关系不会传递给其他链接到 main_app 的目标
target_link_libraries(main_app PRIVATE cuda_task)