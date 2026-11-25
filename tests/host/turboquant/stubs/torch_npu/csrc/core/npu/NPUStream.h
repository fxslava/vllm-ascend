#pragma once
// Declaration-only host syntax seam. No fake Torch tensor or operator types.
namespace c10_npu {
struct HostSyntaxStream { void* stream() const; };
HostSyntaxStream getCurrentNPUStream();
}
