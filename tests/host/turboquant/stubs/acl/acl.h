#pragma once
#include <cstdint>
// The bindings use only this capability query and the opaque stream handle.
using aclrtStream = void*;
constexpr int ACL_SUCCESS = 0;
constexpr int ACL_DEVICE_INFO_VECTOR_CORE_NUM = 0;
int aclGetDeviceCapability(int, int, std::int64_t*);
