# 明确列出源文件
set(UTIL_SOURCES
    src/utils.cu
)
set(UTIL_HEADERS
    include/util/utils.cuh
)

# 创建静态库
add_library(util STATIC
    ${UTIL_SOURCES}
    ${UTIL_HEADERS}
)

# 将 include 目录作为 util 库的公共头文件目录
# PUBLIC 意味着任何链接到 util 的目标都会自动获得这个路径
# 这里的写法已经很规范，无需修改
target_include_directories(util
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)