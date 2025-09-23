# 明确列出源文件，更健壮
set(CUDA_TASK_SOURCES
    src/cuda_task.cu
)
set(CUDA_TASK_HEADERS
    include/cuda_task/cuda_task.cuh
)

# 创建静态库
add_library(cuda_task STATIC 
    ${CUDA_TASK_SOURCES}
    ${CUDA_TASK_HEADERS}
) 

# 指定头文件搜索路径
# PUBLIC: 链接到 cuda_task 的目标也会自动包含这个路径
target_include_directories(cuda_task 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include> # 为未来的安装做准备
)

# 链接依赖库
# PUBLIC: 确保链接到 cuda_task 的目标也能访问 util 的头文件
# CUDA::toolkit: 现代的 CUDA 链接方式
target_link_libraries(cuda_task 
    PUBLIC 
        util
        CUDA::cudart  #链接 CUDA 运行时库 (CUDA Runtime Library) 
        CUDA::cufft   #添加 cuFFT 库链接
)