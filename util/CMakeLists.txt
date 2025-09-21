
# 搜集源文件
file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cu)

# 定义库的目标名
set(TARGET_NAME util)

# 创建静态库
add_library(${TARGET_NAME} STATIC ${srcs})

# 将 include 目录作为 util 库的公共头文件目录
# PUBLIC 意味着任何链接到 util 的目标都会自动获得这个路径
target_include_directories(${TARGET_NAME} 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)