
# 搜集源文件，同时包含 .cu 和 .cuh
file(GLOB_RECURSE srcs CONFIGURE_DEPENDS "src/*.cu" "include/*.cuh")

# 定义可执行文件的目标名
set(TARGET_NAME program_main)

# 创建可执行文件
add_executable(${TARGET_NAME} ${srcs})


# 只添加 program_main 自己的 include 目录
# PRIVATE 因为这是可执行文件，头文件路径不需要传递给别人
target_include_directories(${TARGET_NAME} 
    PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)


# 链接 util 库，PUBLIC/PRIVATE 都可以，这里用 PRIVATE
# CMake 会自动处理好 util 头文件的包含问题
target_link_libraries(${TARGET_NAME} PRIVATE util)