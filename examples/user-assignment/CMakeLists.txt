build_lib_example(
        NAME ns3ai_user_assignment_msg_vec
        SOURCE_FILES use-msg-vec/user_assignment.cc
        LIBRARIES_TO_LINK ${libai} ${libcore} ${libnetwork} ${libmobility} ${libinternet} ${liblte} ${libnetanim}
)

pybind11_add_module(ns3ai_user_assignment_py_vec use-msg-vec/user_assignment_py.cc)

set_target_properties(ns3ai_user_assignment_py_vec PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/use-msg-vec)

# Build Python binding library along with C++ simulation
add_dependencies(ns3ai_user_assignment_msg_vec ns3ai_user_assignment_py_vec)