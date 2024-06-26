if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(TORCH_BUILD_TYPE "debug")
elseif(CMAKE_BUILD_TYPE MATCHES Release OR CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
    set(TORCH_BUILD_TYPE "release")
else()
    set(TORCH_BUILD_TYPE "debug")
endif()

if (WIN32)
    set(LIBTORCH_VERSION "2.1.0") # 2.3.0 crashes, cool
else()
    set(LIBTORCH_VERSION "2.3.0")
endif()

if (WIN32)
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        set(TORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-debug-${LIBTORCH_VERSION}%2Bcpu.zip")
    else()
        set(TORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip")
    endif()
elseif(LINUX)
        set(TORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip")
elseif(APPLE)
        set(TORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-${LIBTORCH_VERSION}.zip")
endif()

find_package(Torch QUIET CONFIG)
if(NOT Torch_FOUND)
    message(STATUS "libtorch not found")
    message(STATUS "Fetching libtorch")
    include(FetchContent)
    FetchContent_Declare(
        libtorch
        URL ${TORCH_URL}
        SOURCE_DIR libtorch)
    FetchContent_GetProperties(libtorch)
    if(NOT libtorch_POPULATED)
        unset(FETCHCONTENT_QUIET CACHE)
        FetchContent_Populate(libtorch)
        list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR}/libtorch)
    endif()
    find_package(Torch ${libtorch_VERSION} EXACT CONFIG REQUIRED)
else()
    message(STATUS "libtorch found")
endif()

# get all .dll files in ${CMAKE_BINARY_DIR}/libtorch/lib
file(GLOB TORCH_DLLS "${CMAKE_BINARY_DIR}/libtorch/lib/*.dll")

# install the dll files to the binary directory for taregt "test"
install(FILES ${TORCH_DLLS} DESTINATION obs-plugins/64bit)
