import struct

import cressim_neo as neo

try:
    import torch
except ImportError as exc:
    raise RuntimeError("This example requires PyTorch to be installed.") from exc


CONSTANT_BUFFER_SHADER = r"""
#include "include/structured_buffer_compat.hlsli"

cbuffer PythonConstantBuffer
{
    float value0;
    float value1;
    float value2;
    float value3;
};

CRESSIM_RW_STRUCTURED_BUFFER(float, g_Output);

[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x != 0u)
    {
        return;
    }

    CRESSIM_SB_STORE(g_Output, 0u, value0);
    CRESSIM_SB_STORE(g_Output, 1u, value1);
    CRESSIM_SB_STORE(g_Output, 2u, value2);
    CRESSIM_SB_STORE(g_Output, 3u, value3);
}
"""


MIXED_CONSTANT_BUFFER_SHADER = r"""
#include "include/structured_buffer_compat.hlsli"

cbuffer PythonMixedConstantBuffer
{
    float value0;
    float value1;
    uint value2;
    float value3;
};

CRESSIM_RW_STRUCTURED_BUFFER(float, g_Output);

[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x != 0u)
    {
        return;
    }

    CRESSIM_SB_STORE(g_Output, 0u, value0);
    CRESSIM_SB_STORE(g_Output, 1u, value1);
    CRESSIM_SB_STORE(g_Output, 2u, float(value2));
    CRESSIM_SB_STORE(g_Output, 3u, value3);
}
"""


def make_tensor(runtime: neo.Runtime, handle: neo.SharedBufferHandle) -> "torch.Tensor":
    desc = neo.SharedBufferTensorDesc()
    desc.shape = [4]
    desc.dtype_code = neo.SharedBufferTensorDTypeCode.Float
    desc.dtype_bits = 32
    desc.dtype_lanes = 1
    return torch.utils.dlpack.from_dlpack(runtime.shared_buffer_to_dlpack(handle, desc))


def create_output_buffer(runtime: neo.Runtime) -> neo.SharedBufferHandle:
    desc = neo.SharedBufferDesc()
    desc.debug_name = "PythonConstantBufferOutput"
    desc.element_stride_bytes = 4
    desc.element_count = 4
    desc.access = neo.SharedBufferAccess.ReadWrite
    desc.bind_flags = (
        neo.SharedBufferBindFlags.ShaderResource | neo.SharedBufferBindFlags.UnorderedAccess
    )
    handle = runtime.create_shared_buffer(desc)
    if not handle.is_valid():
        raise RuntimeError("Failed to create shared output buffer.")
    return handle


def create_pass(
    runtime: neo.Runtime, output_buffer: neo.SharedBufferHandle, constants: tuple[float, ...]
) -> neo.CustomComputePassHandle:
    desc = neo.CustomComputePassDesc()
    desc.debug_name = "PythonConstantBufferTorch"
    desc.shader_source = CONSTANT_BUFFER_SHADER
    desc.thread_group_size_x = 1
    desc.resource_bindings = [neo.CustomComputeResourceBindingDesc()]
    desc.resource_bindings[0].shader_variable_name = "g_Output"
    desc.resource_bindings[0].shared_buffer_handle = output_buffer
    desc.resource_bindings[0].access = neo.CustomComputeResourceAccess.ReadWrite
    desc.constant_buffer_variable_name = "PythonConstantBuffer"
    desc.constant_buffer_size_bytes = 16
    desc.constant_data = list(struct.pack("<4f", *constants))
    desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
    desc.dispatch.group_count_x = 1
    handle = runtime.create_custom_compute_pass(desc)
    if not handle.is_valid():
        raise RuntimeError("Failed to create custom compute pass.")
    return handle


def create_mixed_pass(
    runtime: neo.Runtime, output_buffer: neo.SharedBufferHandle, constants_bytes: bytes
) -> neo.CustomComputePassHandle:
    desc = neo.CustomComputePassDesc()
    desc.debug_name = "PythonMixedConstantBufferTorch"
    desc.shader_source = MIXED_CONSTANT_BUFFER_SHADER
    desc.thread_group_size_x = 1
    desc.resource_bindings = [neo.CustomComputeResourceBindingDesc()]
    desc.resource_bindings[0].shader_variable_name = "g_Output"
    desc.resource_bindings[0].shared_buffer_handle = output_buffer
    desc.resource_bindings[0].access = neo.CustomComputeResourceAccess.ReadWrite
    desc.constant_buffer_variable_name = "PythonMixedConstantBuffer"
    desc.constant_buffer_size_bytes = 16
    desc.constant_data = list(constants_bytes)
    desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
    desc.dispatch.group_count_x = 1
    handle = runtime.create_custom_compute_pass(desc)
    if not handle.is_valid():
        raise RuntimeError("Failed to create mixed custom compute pass.")
    return handle


def execute_and_read(
    runtime: neo.Runtime,
    output_buffer: neo.SharedBufferHandle,
    output_tensor: "torch.Tensor",
    custom_pass: neo.CustomComputePassHandle,
) -> list[float]:
    if not runtime.execute_custom_compute_pass(custom_pass):
        raise RuntimeError("Failed to execute custom compute pass.")
    if not runtime.sync_shared_buffer_to_cuda(output_buffer):
        raise RuntimeError("Failed to synchronize output buffer to CUDA.")
    return output_tensor.detach().cpu().tolist()


def main() -> int:
    runtime = neo.Runtime()
    if not runtime.initialize():
        raise RuntimeError("Failed to initialize runtime.")

    try:
        runtime.prepare()
        if not runtime.upload_world():
            raise RuntimeError("Failed to upload world.")

        output_buffer = create_output_buffer(runtime)
        output_tensor = make_tensor(runtime, output_buffer)

        initial_constants = (1.25, -2.5, 3.75, -4.0)
        updated_constants = (-0.5, 0.25, 8.0, 16.5)
        mixed_initial_constants = (0.75, -1.5, 123, 4.25)
        mixed_updated_constants = (-2.0, 6.5, 500, -0.125)

        custom_pass = create_pass(runtime, output_buffer, initial_constants)
        initial_values = execute_and_read(runtime, output_buffer, output_tensor, custom_pass)
        print("initial:", initial_values)

        if not torch.allclose(
            torch.tensor(initial_values), torch.tensor(initial_constants), atol=1.0e-6, rtol=0.0
        ):
            raise RuntimeError("Initial custom compute constants were not read correctly.")

        if not runtime.update_custom_compute_pass_constants(
            custom_pass, struct.pack("<4f", *updated_constants)
        ):
            raise RuntimeError("Failed to update custom compute pass constants.")

        updated_values = execute_and_read(runtime, output_buffer, output_tensor, custom_pass)
        print("updated:", updated_values)

        if not torch.allclose(
            torch.tensor(updated_values), torch.tensor(updated_constants), atol=1.0e-6, rtol=0.0
        ):
            raise RuntimeError("Updated custom compute constants were not read correctly.")

        mixed_pass = create_mixed_pass(
            runtime, output_buffer, struct.pack("<ffIf", *mixed_initial_constants)
        )
        mixed_initial_values = execute_and_read(runtime, output_buffer, output_tensor, mixed_pass)
        print("mixed initial:", mixed_initial_values)

        if not torch.allclose(
            torch.tensor(mixed_initial_values),
            torch.tensor(
                [
                    mixed_initial_constants[0],
                    mixed_initial_constants[1],
                    float(mixed_initial_constants[2]),
                    mixed_initial_constants[3],
                ]
            ),
            atol=1.0e-6,
            rtol=0.0,
        ):
            raise RuntimeError("Initial mixed custom compute constants were not read correctly.")

        if not runtime.update_custom_compute_pass_constants(
            mixed_pass, struct.pack("<ffIf", *mixed_updated_constants)
        ):
            raise RuntimeError("Failed to update mixed custom compute pass constants.")

        mixed_updated_values = execute_and_read(runtime, output_buffer, output_tensor, mixed_pass)
        print("mixed updated:", mixed_updated_values)

        if not torch.allclose(
            torch.tensor(mixed_updated_values),
            torch.tensor(
                [
                    mixed_updated_constants[0],
                    mixed_updated_constants[1],
                    float(mixed_updated_constants[2]),
                    mixed_updated_constants[3],
                ]
            ),
            atol=1.0e-6,
            rtol=0.0,
        ):
            raise RuntimeError("Updated mixed custom compute constants were not read correctly.")

        runtime.destroy_custom_compute_pass(custom_pass)
        runtime.destroy_custom_compute_pass(mixed_pass)
        runtime.destroy_shared_buffer(output_buffer)
    finally:
        runtime.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
