#------------------------------------------------------------------------------
# Include source and build directories
#------------------------------------------------------------------------------
include_directories(${CMAKE_SOURCE_DIR}/include)

#------------------------------------------------------------------------------
# FLAGS for building
#------------------------------------------------------------------------------

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-result -std=c++11")

#------------------------------------------------------------------------------
# Tools
#------------------------------------------------------------------------------

add_library(reader reader.c)


add_executable(recorder2text recorder2text.c)
target_link_libraries(recorder2text reader)
add_dependencies(recorder2text reader)

add_executable(metaops_checker meta_ops_checker.c)
target_link_libraries(metaops_checker reader)
add_dependencies(metaops_checker reader)


add_executable(conflict_detector overlap_conflict.c build_offset_intervals.cpp)
target_link_libraries(conflict_detector reader)
add_dependencies(conflict_detector reader)