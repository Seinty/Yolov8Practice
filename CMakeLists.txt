# CMakeList.txt: проект CMake для TorchTest; включите исходный код и определения,
# укажите здесь логику для конкретного проекта.
#

# Добавьте источник в исполняемый файл этого проекта.
cmake_minimum_required(VERSION 3.29)

project(testtorch)

add_executable (vstorch "main.cpp")

set(CMAKE_PREFIX_PATH "./libtorch")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(Torch REQUIRED)

find_package(OpenCV  REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(vstorch ${TORCH_LIBRARIES} ${OpenCV_LIBS})
include_directories(${TORCH_INCLUDE_DIRS})

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET vstorch
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:vstorch>)
endif (MSVC)


if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET vstorch PROPERTY CXX_STANDARD 20)
endif()

# TODO: Добавьте тесты и целевые объекты, если это необходимо.
