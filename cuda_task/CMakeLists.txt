
# 搜集源文件，同时包含 .cu 和 .cuh
file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cu include/*.cuh)

# 定义可执行文件的目标名
set(TARGET_NAME cuda_task)

# 创建静态库
add_library(cuda_task STATIC ${srcs}) 


# 保留原有的头文件路径配置
target_include_directories(${TARGET_NAME} 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)


# 链接 util 库，PUBLIC/PRIVATE 都可以，这里用 PRIVATE
# CMake 会自动处理好 util 头文件的包含问题
target_link_libraries(${TARGET_NAME} PUBLIC util  ${CUDA_LIBRARIES})