#include "user_assignment.h"

#include <ns3/ai-module.h>

#include <iostream>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(ns3::Ns3AiMsgInterfaceImpl<UserAssignmentEnv, UserAssignmentAct>::Cpp2PyMsgVector);
PYBIND11_MAKE_OPAQUE(ns3::Ns3AiMsgInterfaceImpl<UserAssignmentEnv, UserAssignmentAct>::Py2CppMsgVector);

PYBIND11_MODULE(ns3ai_user_assignment_py_vec, m)
{
    py::class_<UserAssignmentEnv>(m, "PyEnvStruct")
        .def_readwrite("done", &UserAssignmentEnv::done)
        .def_readwrite("timeSec", &UserAssignmentEnv::timeSec)
        .def_readwrite("ueId", &UserAssignmentEnv::ueId)
        .def_readwrite("numUes", &UserAssignmentEnv::numUes)
        .def_readwrite("numEnbs", &UserAssignmentEnv::numEnbs)
        .def_readwrite("ux", &UserAssignmentEnv::ux)
        .def_readwrite("uy", &UserAssignmentEnv::uy)
        .def_readwrite("vx", &UserAssignmentEnv::vx)
        .def_readwrite("vy", &UserAssignmentEnv::vy)
        .def_readwrite("reward", &UserAssignmentEnv::reward)
        .def_readwrite("totalThroughputMbps", &UserAssignmentEnv::totalThroughputMbps)
        .def_readwrite("loadStd", &UserAssignmentEnv::loadStd)
        .def_readwrite("handovers", &UserAssignmentEnv::handovers)
        .def_readwrite("outages", &UserAssignmentEnv::outages)
        .def_readwrite("step", &UserAssignmentEnv::step)
        .def_readwrite("servingCell", &UserAssignmentEnv::servingCell)
        .def_readwrite("targetBestCell", &UserAssignmentEnv::targetBestCell)
        .def_readwrite("servingSinr", &UserAssignmentEnv::servingSinr)
        .def(
            "get_load",
            [](const UserAssignmentEnv& env, uint32_t i) {
                if (i >= MAX_ENBS)
                {
                    throw std::out_of_range("Load index out of range");
                }
                return env.load[i];
            })
        .def(
            "get_capacity",
            [](const UserAssignmentEnv& env, uint32_t i) {
                if (i >= MAX_ENBS)
                {
                    throw std::out_of_range("Capacity index out of range");
                }
                return env.capacity[i];
            })
        .def(
            "get_rsrp",
            [](const UserAssignmentEnv& env, uint32_t i) {
                if (i >= MAX_ENBS)
                {
                    throw std::out_of_range("RSRP index out of range");
                }
                return env.rsrp[i];
            });

    py::class_<UserAssignmentAct>(m, "PyActStruct")
        .def(py::init<>())
        .def_readwrite("predictedCell", &UserAssignmentAct::predictedCell);

    py::class_<ns3::Ns3AiMsgInterfaceImpl<UserAssignmentEnv, UserAssignmentAct>::Cpp2PyMsgVector>(m, "PyEnvVector")
        .def(
            "resize",
            static_cast<void (ns3::Ns3AiMsgInterfaceImpl<UserAssignmentEnv, UserAssignmentAct>::Cpp2PyMsgVector::*)(
                ns3::Ns3AiMsgInterfaceImpl<UserAssignmentEnv, UserAssignmentAct>::Cpp2PyMsgVector::size_type)>(
                &ns3::Ns3AiMsgInterfaceImpl<UserAssignmentEnv, UserAssignmentAct>::Cpp2PyMsgVector::resize))
        .def("__len__", &ns3::Ns3AiMsgInterfaceImpl<UserAssignmentEnv, UserAssignmentAct>::Cpp2PyMsgVector::size)
        .def(
            "__getitem__",
            [](ns3::Ns3AiMsgInterfaceImpl<UserAssignmentEnv, UserAssignmentAct>::Cpp2PyMsgVector& vec,
               uint32_t i) -> UserAssignmentEnv& {
                if (i >= vec.size())
                {
                    std::cerr << "Invalid env index " << i << std::endl;
                    exit(1);
                }
                return vec.at(i);
            },
            py::return_value_policy::reference);

    py::class_<ns3::Ns3AiMsgInterfaceImpl<UserAssignmentEnv, UserAssignmentAct>::Py2CppMsgVector>(m, "PyActVector")
        .def(
            "resize",
            static_cast<void (ns3::Ns3AiMsgInterfaceImpl<UserAssignmentEnv, UserAssignmentAct>::Py2CppMsgVector::*)(
                ns3::Ns3AiMsgInterfaceImpl<UserAssignmentEnv, UserAssignmentAct>::Py2CppMsgVector::size_type)>(
                &ns3::Ns3AiMsgInterfaceImpl<UserAssignmentEnv, UserAssignmentAct>::Py2CppMsgVector::resize))
        .def("__len__", &ns3::Ns3AiMsgInterfaceImpl<UserAssignmentEnv, UserAssignmentAct>::Py2CppMsgVector::size)
        .def(
            "__getitem__",
            [](ns3::Ns3AiMsgInterfaceImpl<UserAssignmentEnv, UserAssignmentAct>::Py2CppMsgVector& vec,
               uint32_t i) -> UserAssignmentAct& {
                if (i >= vec.size())
                {
                    std::cerr << "Invalid act index " << i << std::endl;
                    exit(1);
                }
                return vec.at(i);
            },
            py::return_value_policy::reference);

    py::class_<ns3::Ns3AiMsgInterfaceImpl<UserAssignmentEnv, UserAssignmentAct>>(m, "Ns3AiMsgInterfaceImpl")
        .def(py::init<bool,
                      bool,
                      bool,
                      uint32_t,
                      const char*,
                      const char*,
                      const char*,
                      const char*>())
        .def("PyRecvBegin", &ns3::Ns3AiMsgInterfaceImpl<UserAssignmentEnv, UserAssignmentAct>::PyRecvBegin)
        .def("PyRecvEnd", &ns3::Ns3AiMsgInterfaceImpl<UserAssignmentEnv, UserAssignmentAct>::PyRecvEnd)
        .def("PySendBegin", &ns3::Ns3AiMsgInterfaceImpl<UserAssignmentEnv, UserAssignmentAct>::PySendBegin)
        .def("PySendEnd", &ns3::Ns3AiMsgInterfaceImpl<UserAssignmentEnv, UserAssignmentAct>::PySendEnd)
        .def("PyGetFinished", &ns3::Ns3AiMsgInterfaceImpl<UserAssignmentEnv, UserAssignmentAct>::PyGetFinished)
        .def("GetCpp2PyVector",
             &ns3::Ns3AiMsgInterfaceImpl<UserAssignmentEnv, UserAssignmentAct>::GetCpp2PyVector,
             py::return_value_policy::reference)
        .def("GetPy2CppVector",
             &ns3::Ns3AiMsgInterfaceImpl<UserAssignmentEnv, UserAssignmentAct>::GetPy2CppVector,
             py::return_value_policy::reference);
}