from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from . import _cressim_neo as neo

try:
    import torch
except ImportError as exc:
    raise RuntimeError("cressim_neo.cartpole requires PyTorch to be installed.") from exc


try:
    import numpy as np
except ImportError:
    np = None

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None


_CARTPOLE_RGB_SHADER = r"""
#include "include/structured_buffer_compat.hlsli"

Texture2DArray<float4> g_ColorTarget;
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_ColorObservation);

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint width = 0u;
    uint height = 0u;
    uint layers = 0u;
    g_ColorTarget.GetDimensions(width, height, layers);

    const uint x = dispatchThreadID.x;
    const uint y = dispatchThreadID.y;
    const uint envIndex = dispatchThreadID.z;
    if (x >= width || y >= height || envIndex >= layers)
    {
        return;
    }

    const uint pixelIndex = envIndex * width * height + y * width + x;
    const float4 color = saturate(g_ColorTarget.Load(int4(int(x), int(y), int(envIndex), 0)));
    CRESSIM_SB_STORE(g_ColorObservation, pixelIndex, color);
}
"""


_PRE_PHYSICS_SHADER = r"""
#include "include/structured_buffer_compat.hlsli"
#include "include/physics/rigid/physics_rigid_types.hlsli"

cbuffer CartpolePrePhysicsConstants
{
    float actionScale;
    float padding0;
    float padding1;
    float padding2;
};

CRESSIM_STRUCTURED_BUFFER(float, g_Actions);
CRESSIM_STRUCTURED_BUFFER(uint, g_SliderJointIndices);
CRESSIM_RW_STRUCTURED_BUFFER(GpuSliderJoint, g_SliderJoints);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint envIndex = dispatchThreadID.x;
    uint envCount = 0u;
    uint stride = 0u;
    g_Actions.GetDimensions(envCount, stride);
    if (envIndex >= envCount)
    {
        return;
    }

    const uint jointIndex = CRESSIM_SB_LOAD(g_SliderJointIndices, envIndex);
    uint jointCount = 0u;
    uint jointStride = 0u;
    g_SliderJoints.GetDimensions(jointCount, jointStride);
    if (jointIndex >= jointCount)
    {
        return;
    }

    GpuSliderJoint joint = CRESSIM_SB_LOAD(g_SliderJoints, jointIndex);
    joint.driveTargetParams.z = CRESSIM_SB_LOAD(g_Actions, envIndex) * actionScale;
    CRESSIM_SB_STORE(g_SliderJoints, jointIndex, joint);
}
"""


_POST_PHYSICS_SHADER = r"""
#include "include/structured_buffer_compat.hlsli"
#include "include/physics/core/physics_math.hlsli"
#include "include/physics/rigid/physics_rigid_joint_solver_shared.hlsli"
#include "include/physics/rigid/physics_rigid_types.hlsli"

cbuffer CartpolePostPhysicsConstants
{
    float cartLimit;
    float poleAngleLimit;
    uint maxEpisodeSteps;
    float padding0;
};

CRESSIM_STRUCTURED_BUFFER(uint, g_BaseBodyIndices);
CRESSIM_STRUCTURED_BUFFER(uint, g_CartBodyIndices);
CRESSIM_STRUCTURED_BUFFER(uint, g_PoleBodyIndices);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyLinearVelocities);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyAngularVelocities);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_Observations);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_Rewards);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_Terminated);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_Truncated);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_EpisodeSteps);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint envIndex = dispatchThreadID.x;
    uint envCount = 0u;
    uint obsStride = 0u;
    g_EpisodeSteps.GetDimensions(envCount, obsStride);
    if (envIndex >= envCount)
    {
        return;
    }

    const uint baseIndex = CRESSIM_SB_LOAD(g_BaseBodyIndices, envIndex);
    const uint cartIndex = CRESSIM_SB_LOAD(g_CartBodyIndices, envIndex);
    const uint poleIndex = CRESSIM_SB_LOAD(g_PoleBodyIndices, envIndex);

    const float4 basePosInvMass = CRESSIM_SB_LOAD(g_RigidBodyPositionsInvMass, baseIndex);
    const float4 cartPosInvMass = CRESSIM_SB_LOAD(g_RigidBodyPositionsInvMass, cartIndex);
    const float4 baseOrientation = QuaternionNormalize(CRESSIM_SB_LOAD(g_RigidBodyOrientations, baseIndex));
    const float4 cartOrientation = QuaternionNormalize(CRESSIM_SB_LOAD(g_RigidBodyOrientations, cartIndex));
    const float4 poleOrientation = QuaternionNormalize(CRESSIM_SB_LOAD(g_RigidBodyOrientations, poleIndex));
    const float3 baseLinearVelocity = CRESSIM_SB_LOAD(g_RigidBodyLinearVelocities, baseIndex).xyz;
    const float3 cartLinearVelocity = CRESSIM_SB_LOAD(g_RigidBodyLinearVelocities, cartIndex).xyz;
    const float3 baseAngularVelocity = CRESSIM_SB_LOAD(g_RigidBodyAngularVelocities, baseIndex).xyz;
    const float3 cartAngularVelocity = CRESSIM_SB_LOAD(g_RigidBodyAngularVelocities, cartIndex).xyz;
    const float3 poleAngularVelocityWorld = CRESSIM_SB_LOAD(g_RigidBodyAngularVelocities, poleIndex).xyz;

    const float3 sliderAxis = SafeNormalize(
        QuaternionRotate(baseOrientation, float3(1.0, 0.0, 0.0)),
        float3(1.0, 0.0, 0.0));
    const float3 hingeAxis = SafeNormalize(
        QuaternionRotate(cartOrientation, float3(0.0, 0.0, 1.0)),
        float3(0.0, 0.0, 1.0));

    const float3 cartDelta = cartPosInvMass.xyz - basePosInvMass.xyz;
    const float cartPosition = dot(cartDelta, sliderAxis);
    const float cartVelocity = dot(cartLinearVelocity - baseLinearVelocity, sliderAxis);

    float3 referenceUp = float3(0.0, 1.0, 0.0) - hingeAxis * dot(float3(0.0, 1.0, 0.0), hingeAxis);
    referenceUp = SafeNormalize(referenceUp, ChoosePerpendicular(hingeAxis));
    const float3 referenceRight = SafeNormalize(cross(hingeAxis, referenceUp), float3(1.0, 0.0, 0.0));
    const float3 poleUp = SafeNormalize(QuaternionRotate(poleOrientation, float3(0.0, 1.0, 0.0)),
                                        referenceUp);
    const float3 planarPoleUp = SafeNormalize(poleUp - hingeAxis * dot(poleUp, hingeAxis), referenceUp);
    const float poleAngle = atan2(dot(planarPoleUp, referenceRight), dot(planarPoleUp, referenceUp));
    const float poleAngularVelocity = dot(poleAngularVelocityWorld - cartAngularVelocity, hingeAxis);

    const uint nextEpisodeStep = CRESSIM_SB_LOAD(g_EpisodeSteps, envIndex) + 1u;
    const uint terminated =
        (abs(cartPosition) > cartLimit || abs(poleAngle) > poleAngleLimit) ? 1u : 0u;
    const uint truncated = nextEpisodeStep >= maxEpisodeSteps ? 1u : 0u;
    const float reward = 1.0f;
    const uint obsBase = envIndex * 4u;
    CRESSIM_SB_STORE(g_Observations, obsBase + 0u, cartPosition);
    CRESSIM_SB_STORE(g_Observations, obsBase + 1u, cartVelocity);
    CRESSIM_SB_STORE(g_Observations, obsBase + 2u, poleAngle);
    CRESSIM_SB_STORE(g_Observations, obsBase + 3u, poleAngularVelocity);
    CRESSIM_SB_STORE(g_Rewards, envIndex, reward);
    CRESSIM_SB_STORE(g_Terminated, envIndex, terminated);
    CRESSIM_SB_STORE(g_Truncated, envIndex, truncated);
    CRESSIM_SB_STORE(g_EpisodeSteps, envIndex, nextEpisodeStep);
}
"""


_RESET_SHADER = r"""
#include "include/structured_buffer_compat.hlsli"
#include "include/physics/core/physics_math.hlsli"
#include "include/physics/rigid/physics_rigid_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(uint, g_ResetMask);
CRESSIM_STRUCTURED_BUFFER(float, g_ResetState);
CRESSIM_STRUCTURED_BUFFER(uint, g_BaseBodyIndices);
CRESSIM_STRUCTURED_BUFFER(uint, g_CartBodyIndices);
CRESSIM_STRUCTURED_BUFFER(uint, g_PoleBodyIndices);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_RigidBodyPositionsInvMass);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_RigidBodyOrientations);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_RigidBodyLinearVelocities);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_RigidBodyAngularVelocities);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_Rewards);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_Terminated);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_Truncated);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_EpisodeSteps);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_Observations);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint envIndex = dispatchThreadID.x;
    uint envCount = 0u;
    uint stride = 0u;
    g_ResetMask.GetDimensions(envCount, stride);
    if (envIndex >= envCount || CRESSIM_SB_LOAD(g_ResetMask, envIndex) == 0u)
    {
        return;
    }

    const uint baseIndex = CRESSIM_SB_LOAD(g_BaseBodyIndices, envIndex);
    const uint cartIndex = CRESSIM_SB_LOAD(g_CartBodyIndices, envIndex);
    const uint poleIndex = CRESSIM_SB_LOAD(g_PoleBodyIndices, envIndex);

    const float cartPosition = CRESSIM_SB_LOAD(g_ResetState, envIndex * 4u + 0u);
    const float cartVelocity = CRESSIM_SB_LOAD(g_ResetState, envIndex * 4u + 1u);
    const float poleAngle = CRESSIM_SB_LOAD(g_ResetState, envIndex * 4u + 2u);
    const float poleAngularVelocityScalar = CRESSIM_SB_LOAD(g_ResetState, envIndex * 4u + 3u);

    const float4 basePosInvMass = CRESSIM_SB_LOAD(g_RigidBodyPositionsInvMass, baseIndex);
    const float4 baseOrientation =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_RigidBodyOrientations, baseIndex));
    const float3 baseLinearVelocity = CRESSIM_SB_LOAD(g_RigidBodyLinearVelocities, baseIndex).xyz;
    const float3 sliderAxis = QuaternionRotate(baseOrientation, float3(1.0, 0.0, 0.0));
    const float3 hingeAxis = QuaternionRotate(baseOrientation, float3(0.0, 0.0, 1.0));

    const float3 cartPositionWorld = basePosInvMass.xyz + sliderAxis * cartPosition;
    const float4 cartOrientation = baseOrientation;
    const float4 poleOrientation =
        QuaternionNormalize(QuaternionMul(QuaternionFromRotationVector(hingeAxis * poleAngle),
                                          cartOrientation));

    const float3 cartAnchorWorld = QuaternionRotate(cartOrientation, float3(0.0, 0.10, 0.0));
    const float3 poleAnchorWorld = QuaternionRotate(poleOrientation, float3(0.0, -0.50, 0.0));
    const float3 hingePointWorld = cartPositionWorld + cartAnchorWorld;
    const float3 polePositionWorld = hingePointWorld - poleAnchorWorld;

    const float3 cartLinearVelocityWorld = baseLinearVelocity + sliderAxis * cartVelocity;
    const float3 cartAngularVelocityWorld = float3(0.0, 0.0, 0.0);
    const float3 poleAngularVelocityWorld = cartAngularVelocityWorld + hingeAxis * poleAngularVelocityScalar;
    const float3 hingePointVelocity =
        cartLinearVelocityWorld + cross(cartAngularVelocityWorld, cartAnchorWorld);
    const float3 poleLinearVelocityWorld =
        hingePointVelocity - cross(poleAngularVelocityWorld, poleAnchorWorld);

    const float cartInvMass = CRESSIM_SB_LOAD(g_RigidBodyPositionsInvMass, cartIndex).w;
    const float poleInvMass = CRESSIM_SB_LOAD(g_RigidBodyPositionsInvMass, poleIndex).w;
    CRESSIM_SB_STORE(g_RigidBodyPositionsInvMass, cartIndex,
                     float4(cartPositionWorld, cartInvMass));
    CRESSIM_SB_STORE(g_RigidBodyPositionsInvMass, poleIndex,
                     float4(polePositionWorld, poleInvMass));
    CRESSIM_SB_STORE(g_RigidBodyOrientations, cartIndex, cartOrientation);
    CRESSIM_SB_STORE(g_RigidBodyOrientations, poleIndex, poleOrientation);
    CRESSIM_SB_STORE(g_RigidBodyLinearVelocities, cartIndex,
                     float4(cartLinearVelocityWorld, 0.0));
    CRESSIM_SB_STORE(g_RigidBodyLinearVelocities, poleIndex,
                     float4(poleLinearVelocityWorld, 0.0));
    CRESSIM_SB_STORE(g_RigidBodyAngularVelocities, cartIndex,
                     float4(cartAngularVelocityWorld, 0.0));
    CRESSIM_SB_STORE(g_RigidBodyAngularVelocities, poleIndex,
                     float4(poleAngularVelocityWorld, 0.0));

    CRESSIM_SB_STORE(g_Rewards, envIndex, 0.0f);
    CRESSIM_SB_STORE(g_Terminated, envIndex, 0u);
    CRESSIM_SB_STORE(g_Truncated, envIndex, 0u);
    CRESSIM_SB_STORE(g_EpisodeSteps, envIndex, 0u);

    const uint obsBase = envIndex * 4u;
    CRESSIM_SB_STORE(g_Observations, obsBase + 0u, cartPosition);
    CRESSIM_SB_STORE(g_Observations, obsBase + 1u, cartVelocity);
    CRESSIM_SB_STORE(g_Observations, obsBase + 2u, poleAngle);
    CRESSIM_SB_STORE(g_Observations, obsBase + 3u, poleAngularVelocityScalar);
}
"""


@dataclass(frozen=True)
class _CartpoleIds:
    base_entity: int
    cart_entity: int
    pole_entity: int
    slider_joint_id: int
    hinge_joint_id: int


def _normalize3(x: float, y: float, z: float) -> tuple[float, float, float]:
    length = math.sqrt(x * x + y * y + z * z)
    if length <= 1.0e-8:
        return (1.0, 0.0, 0.0)
    return (x / length, y / length, z / length)


def _quaternion_from_basis(
    x_axis: tuple[float, float, float],
    y_axis: tuple[float, float, float],
    z_axis: tuple[float, float, float],
) -> neo.Quaternion:
    m00, m01, m02 = x_axis[0], y_axis[0], z_axis[0]
    m10, m11, m12 = x_axis[1], y_axis[1], z_axis[1]
    m20, m21, m22 = x_axis[2], y_axis[2], z_axis[2]
    trace = m00 + m11 + m22
    q = neo.Quaternion()
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q.w = 0.25 * s
        q.x = (m21 - m12) / s
        q.y = (m02 - m20) / s
        q.z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        q.w = (m21 - m12) / s
        q.x = 0.25 * s
        q.y = (m01 + m10) / s
        q.z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        q.w = (m02 - m20) / s
        q.x = (m01 + m10) / s
        q.y = 0.25 * s
        q.z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        q.w = (m10 - m01) / s
        q.x = (m02 + m20) / s
        q.y = (m12 + m21) / s
        q.z = 0.25 * s

    length = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    if length <= 1.0e-8:
        q.x = 0.0
        q.y = 0.0
        q.z = 0.0
        q.w = 1.0
        return q
    q.x /= length
    q.y /= length
    q.z /= length
    q.w /= length
    return q


def _make_joint_frame_rotation(axis_x: tuple[float, float, float]) -> neo.Quaternion:
    x = _normalize3(*axis_x)
    reference = (1.0, 0.0, 0.0)
    if abs(x[0] * reference[0] + x[1] * reference[1] + x[2] * reference[2]) > 0.99:
        reference = (0.0, 1.0, 0.0)
    y = _normalize3(
        x[1] * reference[2] - x[2] * reference[1],
        x[2] * reference[0] - x[0] * reference[2],
        x[0] * reference[1] - x[1] * reference[0],
    )
    z = _normalize3(
        x[1] * y[2] - x[2] * y[1],
        x[2] * y[0] - x[0] * y[2],
        x[0] * y[1] - x[1] * y[0],
    )
    return _quaternion_from_basis(x, y, z)


def _make_tensor(
    runtime: neo.Runtime,
    handle: neo.SharedBufferHandle,
    shape: list[int],
    dtype_code: neo.SharedBufferTensorDTypeCode,
) -> "torch.Tensor":
    desc = neo.SharedBufferTensorDesc()
    desc.shape = shape
    desc.dtype_code = dtype_code
    desc.dtype_bits = 32
    desc.dtype_lanes = 1
    return torch.utils.dlpack.from_dlpack(runtime.shared_buffer_to_dlpack(handle, desc))

def _create_buffer(
    runtime: neo.Runtime,
    name: str,
    count: int,
    dtype_code: neo.SharedBufferTensorDTypeCode,
    *,
    element_stride_bytes: int = 4,
    shape: list[int] | None = None,
) -> tuple[neo.SharedBufferHandle, "torch.Tensor"]:
    desc = neo.SharedBufferDesc()
    desc.debug_name = name
    desc.element_stride_bytes = element_stride_bytes
    desc.element_count = count
    desc.access = neo.SharedBufferAccess.ReadWrite
    desc.bind_flags = (
        neo.SharedBufferBindFlags.ShaderResource | neo.SharedBufferBindFlags.UnorderedAccess
    )
    handle = runtime.create_shared_buffer(desc)
    if not handle.is_valid():
        raise RuntimeError(f"Failed to create shared buffer '{name}'.")
    return handle, _make_tensor(runtime, handle, shape or [count], dtype_code)


class CartpoleTorchVectorEnv:
    OBSERVATION_DIM = 4

    def __init__(
        self,
        env_count: int = 64,
        max_episode_steps: int = 500,
        action_scale: float = 8.0,
        cart_limit: float = 2.4,
        pole_angle_limit_radians: float = 12.0 * math.pi / 180.0,
        reset_cart_position_range: float = 0.05,
        reset_cart_velocity_range: float = 0.05,
        reset_pole_angle_range_radians: float = 0.05,
        reset_pole_angular_velocity_range: float = 0.05,
        image_width: int = 64,
        image_height: int = 64,
    ) -> None:
        self.env_count = env_count
        self.max_episode_steps = max_episode_steps
        self.action_scale = action_scale
        self.cart_limit = cart_limit
        self.pole_angle_limit_radians = pole_angle_limit_radians
        self.reset_cart_position_range = reset_cart_position_range
        self.reset_cart_velocity_range = reset_cart_velocity_range
        self.reset_pole_angle_range_radians = reset_pole_angle_range_radians
        self.reset_pole_angular_velocity_range = reset_pole_angular_velocity_range
        self.image_width = image_width
        self.image_height = image_height
        self._frame = neo.FrameContext()
        self._frame.delta_seconds = 1.0 / 60.0
        self._frame.frame_index = 0
        self._frame.time_seconds = 0.0

        config = neo.RuntimeConfig()
        config.gpu_device_desc.preferred_backend = neo.GpuBackend.Vulkan
        config.gpu_device_desc.enable_validation = False
        config.physics_desc.enable_blocking_readback = False
        config.scene_layout.env_count = env_count
        config.scene_layout.max_renderable_objects_per_env = 8
        config.scene_layout.max_lights_per_env = 2
        config.scene_layout.max_cameras_per_env = 1

        self.runtime = neo.Runtime()
        if not self.runtime.initialize(config):
            raise RuntimeError("Failed to initialize cartpole runtime.")

        self._initialize_rgb_observation_resources()
        self._authored_ids = self._author_scene(self.runtime.world(), env_count)
        self.runtime.prepare()
        self._rigid_mapping = self.runtime.get_prepared_rigid_layout_mapping()
        self._joint_mapping = self.runtime.get_prepared_joint_layout_mapping()
        if not self.runtime.upload_world():
            self.close()
            raise RuntimeError("Failed to upload prepared cartpole world.")

        self._create_shared_buffers()
        self._populate_lookup_buffers()
        self._create_custom_passes()
        self.reset()

    def _initialize_rgb_observation_resources(self) -> None:
        resources = self.runtime.resources()

        target_desc = neo.GpuRenderTargetDesc()
        target_desc.width = self.image_width
        target_desc.height = self.image_height
        target_desc.array_size = self.env_count
        target_desc.color = True
        target_desc.depth = True
        target_desc.color_format = neo.TextureFormat.RGBA8UnormSrgb
        target_desc.layered_rendering = True
        target_desc.shader_readable = True
        target_desc.debug_name = "Cartpole.RgbObservationTarget"
        self._rgb_render_target = self.runtime.create_render_target(target_desc)
        if not self.runtime.is_valid_render_target(self._rgb_render_target):
            raise RuntimeError("Failed to create cartpole RGB render target.")

        self._rgb_cube_mesh = resources.register_mesh(neo.make_cube_mesh(0.5, "Cartpole.RenderCube"))
        self._rgb_ground_mesh = resources.register_mesh(
            neo.make_plane_mesh(4.0, "Cartpole.RenderGround", 2.0)
        )

    def _make_material(
        self,
        debug_name: str,
        base_color: neo.Float3,
        roughness: float,
        emissive: neo.Float3 | None = None,
    ) -> neo.MaterialHandle:
        material_desc = neo.MaterialResourceDesc()
        material_desc.debug_name = debug_name
        material_desc.base_color = base_color
        material_desc.metallic = 0.0
        material_desc.roughness = roughness
        if emissive is not None:
            material_desc.emissive_factor = emissive
        return self.runtime.resources().register_material(material_desc)

    def _author_rgb_render_scene(
        self,
        world: neo.World,
        env_index: int,
        base_entity: int,
        cart_entity: int,
        pole_entity: int,
        z_offset: float,
    ) -> None:
        palette = (
            neo.Float3(0.86, 0.24, 0.20),
            neo.Float3(0.20, 0.76, 0.30),
            neo.Float3(0.18, 0.44, 0.92),
            neo.Float3(0.94, 0.73, 0.18),
        )
        cart_material = self._make_material(
            f"Cartpole.CartMaterial.{env_index}",
            neo.Float3(0.70, 0.72, 0.78),
            0.55,
        )
        pole_material = self._make_material(
            f"Cartpole.PoleMaterial.{env_index}",
            palette[env_index % len(palette)],
            0.35,
        )
        base_material = self._make_material(
            f"Cartpole.BaseMaterial.{env_index}",
            neo.Float3(0.26, 0.30, 0.36),
            0.75,
        )
        ground_material = self._make_material(
            f"Cartpole.GroundMaterial.{env_index}",
            neo.Float3(0.58, 0.60, 0.64),
            0.92,
        )

        base_renderer = neo.MeshRendererComponent()
        base_renderer.mesh = self._rgb_cube_mesh
        base_renderer.material = base_material
        base_renderer.segmentation_id = 100 + env_index
        base_renderer.visible = True
        world.set_mesh_renderer(base_entity, base_renderer)

        cart_renderer = neo.MeshRendererComponent()
        cart_renderer.mesh = self._rgb_cube_mesh
        cart_renderer.material = cart_material
        cart_renderer.segmentation_id = 200 + env_index
        cart_renderer.visible = True
        world.set_mesh_renderer(cart_entity, cart_renderer)

        pole_renderer = neo.MeshRendererComponent()
        pole_renderer.mesh = self._rgb_cube_mesh
        pole_renderer.material = pole_material
        pole_renderer.segmentation_id = 300 + env_index
        pole_renderer.visible = True
        world.set_mesh_renderer(pole_entity, pole_renderer)

        ground_entity = world.create_entity(env_index)
        ground_transform = neo.TransformComponent()
        ground_transform.world_transform.position = neo.Float3(0.0, 0.0, z_offset)
        world.set_transform(ground_entity, ground_transform)
        ground_renderer = neo.MeshRendererComponent()
        ground_renderer.mesh = self._rgb_ground_mesh
        ground_renderer.material = ground_material
        ground_renderer.segmentation_id = 400 + env_index
        ground_renderer.visible = True
        world.set_mesh_renderer(ground_entity, ground_renderer)

        light_entity = world.create_entity(env_index)
        light = neo.DirectionalLightComponent()
        light.direction = neo.Float3(-0.45, -1.0, 0.35)
        light.color = neo.Float3(1.0, 1.0, 1.0)
        light.intensity = 7.0
        light.casts_shadows = False
        world.set_directional_light(light_entity, light)

        camera_entity = world.create_entity(env_index)
        camera_transform = neo.TransformComponent()
        camera_transform.world_transform.position = neo.Float3(0.0, 1.35, z_offset - 6.0)
        world.set_transform(camera_entity, camera_transform)

        camera = neo.CameraComponent()
        camera.product = neo.CameraProduct.ColorDepth
        camera.vertical_fov_degrees = 42.0
        camera.output.mode = neo.RenderOutputMode.ExplicitSurface
        camera.output.binding = neo.GpuRenderTargetBinding()
        camera.output.binding.target = self._rgb_render_target
        camera.output.binding.first_layer = env_index
        camera.output.binding.layer_count = 1
        camera.output_width = self.image_width
        camera.output_height = self.image_height
        camera.clear_color = True
        camera.clear_depth = True
        camera.clear_color_value = neo.Float4(0.04, 0.05, 0.08, 1.0)
        world.set_camera(camera_entity, camera)

    def _author_scene(self, world: neo.World, env_count: int) -> list[_CartpoleIds]:
        authored: list[_CartpoleIds] = []
        joint_axis_z = _make_joint_frame_rotation((0.0, 0.0, 1.0))
        joint_axis_x = _make_joint_frame_rotation((1.0, 0.0, 0.0))
        cart_half = neo.Float4(0.18, 0.10, 0.10, 0.0)
        pole_half = neo.Float4(0.05, 0.50, 0.05, 0.0)
        for env_index in range(env_count):
            z_offset = env_index * 2.5
            base_entity = world.create_entity(env_index)
            base_transform = neo.TransformComponent()
            base_transform.world_transform.position = neo.Float3(0.0, 0.5, z_offset)
            base_transform.world_transform.scale = neo.Float3(0.30, 0.30, 0.30)
            world.set_transform(base_entity, base_transform)
            base_body = neo.RigidBodyComponent()
            base_body.body_type = neo.RigidBodyType.Static
            base_body.inverse_mass = 0.0
            base_body.simulated = True
            world.set_rigid_body(base_entity, base_body)
            base_collider = neo.ColliderComponent()
            base_collider.shape_type = neo.ColliderShapeType.Box
            base_collider.shape_params = neo.Float4(0.15, 0.15, 0.15, 0.0)
            world.add_collider(base_entity, base_collider)

            cart_entity = world.create_entity(env_index)
            cart_transform = neo.TransformComponent()
            cart_transform.world_transform.position = neo.Float3(0.0, 0.5, z_offset)
            cart_transform.world_transform.scale = neo.Float3(0.36, 0.20, 0.20)
            world.set_transform(cart_entity, cart_transform)
            cart_body = neo.RigidBodyComponent()
            cart_body.body_type = neo.RigidBodyType.Dynamic
            cart_body.inverse_mass = 1.0
            cart_body.inverse_inertia_local = neo.Float3(2.0, 2.0, 2.0)
            cart_body.simulated = True
            world.set_rigid_body(cart_entity, cart_body)
            cart_collider = neo.ColliderComponent()
            cart_collider.shape_type = neo.ColliderShapeType.Box
            cart_collider.shape_params = cart_half
            world.add_collider(cart_entity, cart_collider)

            pole_entity = world.create_entity(env_index)
            pole_transform = neo.TransformComponent()
            pole_transform.world_transform.position = neo.Float3(0.0, 1.1, z_offset)
            pole_transform.world_transform.scale = neo.Float3(0.10, 1.00, 0.10)
            world.set_transform(pole_entity, pole_transform)
            pole_body = neo.RigidBodyComponent()
            pole_body.body_type = neo.RigidBodyType.Dynamic
            pole_body.inverse_mass = 0.5
            pole_body.inverse_inertia_local = neo.Float3(1.5, 1.5, 1.5)
            pole_body.simulated = True
            world.set_rigid_body(pole_entity, pole_body)
            pole_collider = neo.ColliderComponent()
            pole_collider.shape_type = neo.ColliderShapeType.Box
            pole_collider.shape_params = pole_half
            world.add_collider(pole_entity, pole_collider)

            slider_joint = neo.SliderJointState()
            slider_joint.joint_id = 1000 + env_index
            slider_joint.body_a = base_entity
            slider_joint.body_b = cart_entity
            slider_joint.suppress_connected_body_collisions = True
            slider_joint.drive_mode = neo.RigidJointDriveMode.TargetVelocity
            slider_joint.local_rotation_a = joint_axis_x
            slider_joint.local_rotation_b = joint_axis_x
            slider_joint.limit_enabled = True
            slider_joint.limit_min = -self.cart_limit
            slider_joint.limit_max = self.cart_limit
            slider_joint.drive_target_velocity = 0.0
            if not world.upsert_slider_joint(slider_joint):
                raise RuntimeError(f"Failed to author slider joint for env {env_index}.")

            hinge_joint = neo.HingeJointState()
            hinge_joint.joint_id = 2000 + env_index
            hinge_joint.body_a = cart_entity
            hinge_joint.body_b = pole_entity
            hinge_joint.suppress_connected_body_collisions = True
            hinge_joint.local_anchor_a = neo.Float3(0.0, 0.10, 0.0)
            hinge_joint.local_anchor_b = neo.Float3(0.0, -0.50, 0.0)
            hinge_joint.local_rotation_a = joint_axis_z
            hinge_joint.local_rotation_b = joint_axis_z
            hinge_joint.limit_enabled = False
            if not world.upsert_hinge_joint(hinge_joint):
                raise RuntimeError(f"Failed to author hinge joint for env {env_index}.")

            self._author_rgb_render_scene(
                world,
                env_index,
                base_entity,
                cart_entity,
                pole_entity,
                z_offset,
            )

            authored.append(
                _CartpoleIds(
                    base_entity=base_entity,
                    cart_entity=cart_entity,
                    pole_entity=pole_entity,
                    slider_joint_id=slider_joint.joint_id,
                    hinge_joint_id=hinge_joint.joint_id,
                )
            )

        return authored

    def _create_shared_buffers(self) -> None:
        self._shared_handles: list[neo.SharedBufferHandle] = []

        self.action_buffer, self.action_tensor = _create_buffer(
            self.runtime, "Cartpole.Actions", self.env_count, neo.SharedBufferTensorDTypeCode.Float
        )
        self._shared_handles.append(self.action_buffer)

        self.observation_buffer, observation_flat = _create_buffer(
            self.runtime,
            "Cartpole.Observations",
            self.env_count * self.OBSERVATION_DIM,
            neo.SharedBufferTensorDTypeCode.Float,
        )
        self._shared_handles.append(self.observation_buffer)
        self.observation_tensor = observation_flat.view(self.env_count, self.OBSERVATION_DIM)

        self.reward_buffer, self.reward_tensor = _create_buffer(
            self.runtime, "Cartpole.Rewards", self.env_count, neo.SharedBufferTensorDTypeCode.Float
        )
        self._shared_handles.append(self.reward_buffer)

        self.terminated_buffer, self.terminated_tensor = _create_buffer(
            self.runtime,
            "Cartpole.Terminated",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.UInt,
        )
        self._shared_handles.append(self.terminated_buffer)

        self.truncated_buffer, self.truncated_tensor = _create_buffer(
            self.runtime,
            "Cartpole.Truncated",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.UInt,
        )
        self._shared_handles.append(self.truncated_buffer)

        self.reset_mask_buffer, self.reset_mask_tensor = _create_buffer(
            self.runtime,
            "Cartpole.ResetMask",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.UInt,
        )
        self._shared_handles.append(self.reset_mask_buffer)

        self.episode_steps_buffer, self.episode_steps_tensor = _create_buffer(
            self.runtime,
            "Cartpole.EpisodeSteps",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.UInt,
        )
        self._shared_handles.append(self.episode_steps_buffer)

        self.reset_state_buffer, reset_state_flat = _create_buffer(
            self.runtime,
            "Cartpole.ResetState",
            self.env_count * self.OBSERVATION_DIM,
            neo.SharedBufferTensorDTypeCode.Float,
        )
        self._shared_handles.append(self.reset_state_buffer)
        self.reset_state_tensor = reset_state_flat.view(self.env_count, self.OBSERVATION_DIM)

        self.base_body_buffer, self.base_body_tensor = _create_buffer(
            self.runtime, "Cartpole.BaseBodies", self.env_count, neo.SharedBufferTensorDTypeCode.UInt
        )
        self._shared_handles.append(self.base_body_buffer)
        self.cart_body_buffer, self.cart_body_tensor = _create_buffer(
            self.runtime, "Cartpole.CartBodies", self.env_count, neo.SharedBufferTensorDTypeCode.UInt
        )
        self._shared_handles.append(self.cart_body_buffer)
        self.pole_body_buffer, self.pole_body_tensor = _create_buffer(
            self.runtime, "Cartpole.PoleBodies", self.env_count, neo.SharedBufferTensorDTypeCode.UInt
        )
        self._shared_handles.append(self.pole_body_buffer)
        self.slider_joint_buffer, self.slider_joint_tensor = _create_buffer(
            self.runtime,
            "Cartpole.SliderJoints",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.UInt,
        )
        self._shared_handles.append(self.slider_joint_buffer)
        self.hinge_joint_buffer, self.hinge_joint_tensor = _create_buffer(
            self.runtime,
            "Cartpole.HingeJoints",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.UInt,
        )
        self._shared_handles.append(self.hinge_joint_buffer)

        pixel_count = self.env_count * self.image_width * self.image_height
        self.rgb_observation_buffer, self.rgb_observation_tensor = _create_buffer(
            self.runtime,
            "Cartpole.RgbObservations",
            pixel_count,
            neo.SharedBufferTensorDTypeCode.Float,
            element_stride_bytes=16,
            shape=[self.env_count, self.image_height, self.image_width, 4],
        )
        self._shared_handles.append(self.rgb_observation_buffer)

    def _populate_lookup_buffers(self) -> None:
        rigid_body_slots = {
            entity_id: slot
            for slot, entity_id in enumerate(self._rigid_mapping.rigid_body_entity_ids)
        }
        slider_slots = {
            joint_id: slot
            for slot, joint_id in enumerate(self._joint_mapping.slider_joint_ids)
        }
        hinge_slots = {
            joint_id: slot
            for slot, joint_id in enumerate(self._joint_mapping.hinge_joint_ids)
        }

        device = self.action_tensor.device
        self.base_body_tensor.copy_(
            torch.tensor(
                [rigid_body_slots[item.base_entity] for item in self._authored_ids],
                device=device,
                dtype=self.base_body_tensor.dtype,
            )
        )
        self.cart_body_tensor.copy_(
            torch.tensor(
                [rigid_body_slots[item.cart_entity] for item in self._authored_ids],
                device=device,
                dtype=self.cart_body_tensor.dtype,
            )
        )
        self.pole_body_tensor.copy_(
            torch.tensor(
                [rigid_body_slots[item.pole_entity] for item in self._authored_ids],
                device=device,
                dtype=self.pole_body_tensor.dtype,
            )
        )
        self.slider_joint_tensor.copy_(
            torch.tensor(
                [slider_slots[item.slider_joint_id] for item in self._authored_ids],
                device=device,
                dtype=self.slider_joint_tensor.dtype,
            )
        )
        self.hinge_joint_tensor.copy_(
            torch.tensor(
                [hinge_slots[item.hinge_joint_id] for item in self._authored_ids],
                device=device,
                dtype=self.hinge_joint_tensor.dtype,
            )
        )
        self.reset_mask_tensor.zero_()
        self.episode_steps_tensor.zero_()
        self.reward_tensor.zero_()
        self.terminated_tensor.zero_()
        self.truncated_tensor.zero_()
        self.action_tensor.zero_()
        self.observation_tensor.zero_()
        self.reset_state_tensor.zero_()

        for handle in (
            self.base_body_buffer,
            self.cart_body_buffer,
            self.pole_body_buffer,
            self.slider_joint_buffer,
            self.hinge_joint_buffer,
            self.reset_mask_buffer,
            self.episode_steps_buffer,
            self.reward_buffer,
            self.terminated_buffer,
            self.truncated_buffer,
            self.action_buffer,
            self.observation_buffer,
            self.reset_state_buffer,
        ):
            if not self.runtime.sync_shared_buffer_from_cuda(handle):
                raise RuntimeError("Failed to upload initial cartpole shared buffers.")

    def _create_custom_passes(self) -> None:
        pre_desc = neo.CustomComputePassDesc()
        pre_desc.debug_name = "Cartpole.PrePhysicsControl"
        pre_desc.shader_source = _PRE_PHYSICS_SHADER
        pre_desc.thread_group_size_x = 64
        pre_desc.resource_bindings = [neo.CustomComputeResourceBindingDesc() for _ in range(3)]
        pre_desc.resource_bindings[0].shader_variable_name = "g_Actions"
        pre_desc.resource_bindings[0].shared_buffer_handle = self.action_buffer
        pre_desc.resource_bindings[0].access = neo.CustomComputeResourceAccess.ReadOnly
        pre_desc.resource_bindings[1].shader_variable_name = "g_SliderJointIndices"
        pre_desc.resource_bindings[1].shared_buffer_handle = self.slider_joint_buffer
        pre_desc.resource_bindings[1].access = neo.CustomComputeResourceAccess.ReadOnly
        pre_desc.resource_bindings[2].shader_variable_name = "g_SliderJoints"
        pre_desc.resource_bindings[2].resource_key = "joint.slider"
        pre_desc.resource_bindings[2].access = neo.CustomComputeResourceAccess.ReadWrite
        pre_desc.constant_buffer_variable_name = "CartpolePrePhysicsConstants"
        pre_desc.constant_buffer_size_bytes = 16
        pre_desc.constant_data = list(struct.pack("<4f", self.action_scale, 0.0, 0.0, 0.0))
        pre_desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
        pre_desc.dispatch.group_count_x = (self.env_count + 63) // 64
        self._pre_pass = self.runtime.create_custom_compute_pass(pre_desc)
        if not self._pre_pass.is_valid():
            raise RuntimeError("Failed to create cartpole pre-physics pass.")
        if not self.runtime.update_custom_compute_pass_constants(
            self._pre_pass, struct.pack("<4f", self.action_scale, 0.0, 0.0, 0.0)
        ):
            raise RuntimeError("Failed to upload cartpole pre-physics constants.")

        post_desc = neo.CustomComputePassDesc()
        post_desc.debug_name = "Cartpole.PostPhysicsObservations"
        post_desc.shader_source = _POST_PHYSICS_SHADER
        post_desc.thread_group_size_x = 64
        post_desc.resource_bindings = [neo.CustomComputeResourceBindingDesc() for _ in range(12)]
        bindings = post_desc.resource_bindings
        binding_specs = [
            ("g_BaseBodyIndices", self.base_body_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_CartBodyIndices", self.cart_body_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_PoleBodyIndices", self.pole_body_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_RigidBodyPositionsInvMass", None, "rigid.positions", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_RigidBodyOrientations", None, "rigid.orientations", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_RigidBodyLinearVelocities", None, "rigid.linear_velocities", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_RigidBodyAngularVelocities", None, "rigid.angular_velocities", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_Observations", self.observation_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
            ("g_Rewards", self.reward_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
            ("g_Terminated", self.terminated_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
            ("g_Truncated", self.truncated_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
            ("g_EpisodeSteps", self.episode_steps_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
        ]
        for binding, (name, handle, key, access) in zip(bindings, binding_specs):
            binding.shader_variable_name = name
            binding.access = access
            if handle is not None:
                binding.shared_buffer_handle = handle
            else:
                binding.resource_key = key
        post_desc.constant_buffer_variable_name = "CartpolePostPhysicsConstants"
        post_desc.constant_buffer_size_bytes = 16
        post_desc.constant_data = list(
            struct.pack(
                "<ffIf",
                self.cart_limit,
                self.pole_angle_limit_radians,
                self.max_episode_steps,
                0.0,
            )
        )
        post_desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
        post_desc.dispatch.group_count_x = (self.env_count + 63) // 64
        self._post_pass = self.runtime.create_custom_compute_pass(post_desc)
        if not self._post_pass.is_valid():
            raise RuntimeError("Failed to create cartpole post-physics pass.")
        if not self.runtime.update_custom_compute_pass_constants(
            self._post_pass,
            struct.pack(
                "<ffIf",
                self.cart_limit,
                self.pole_angle_limit_radians,
                self.max_episode_steps,
                0.0,
            ),
        ):
            raise RuntimeError("Failed to upload cartpole post-physics constants.")

        reset_desc = neo.CustomComputePassDesc()
        reset_desc.debug_name = "Cartpole.Reset"
        reset_desc.shader_source = _RESET_SHADER
        reset_desc.thread_group_size_x = 64
        reset_desc.resource_bindings = [neo.CustomComputeResourceBindingDesc() for _ in range(14)]
        reset_specs = [
            ("g_ResetMask", self.reset_mask_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_ResetState", self.reset_state_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_BaseBodyIndices", self.base_body_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_CartBodyIndices", self.cart_body_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_PoleBodyIndices", self.pole_body_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_RigidBodyPositionsInvMass", None, "rigid.positions", neo.CustomComputeResourceAccess.ReadWrite),
            ("g_RigidBodyOrientations", None, "rigid.orientations", neo.CustomComputeResourceAccess.ReadWrite),
            ("g_RigidBodyLinearVelocities", None, "rigid.linear_velocities", neo.CustomComputeResourceAccess.ReadWrite),
            ("g_RigidBodyAngularVelocities", None, "rigid.angular_velocities", neo.CustomComputeResourceAccess.ReadWrite),
            ("g_Rewards", self.reward_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
            ("g_Terminated", self.terminated_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
            ("g_Truncated", self.truncated_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
            ("g_EpisodeSteps", self.episode_steps_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
            ("g_Observations", self.observation_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
        ]
        for binding, (name, handle, key, access) in zip(reset_desc.resource_bindings, reset_specs):
            binding.shader_variable_name = name
            binding.access = access
            if handle is not None:
                binding.shared_buffer_handle = handle
            else:
                binding.resource_key = key
        reset_desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
        reset_desc.dispatch.group_count_x = (self.env_count + 63) // 64
        self._reset_pass = self.runtime.create_custom_compute_pass(reset_desc)
        if not self._reset_pass.is_valid():
            raise RuntimeError("Failed to create cartpole reset pass.")

        render_desc = neo.CustomComputePassDesc()
        render_desc.debug_name = "Cartpole.RgbObservation"
        render_desc.shader_source = _CARTPOLE_RGB_SHADER
        render_desc.thread_group_size_x = 8
        render_desc.thread_group_size_y = 8
        render_desc.resource_bindings = [neo.CustomComputeResourceBindingDesc() for _ in range(2)]
        render_desc.resource_bindings[0].shader_variable_name = "g_ColorTarget"
        render_desc.resource_bindings[0].render_target_binding = neo.GpuRenderTargetBinding()
        render_desc.resource_bindings[0].render_target_binding.target = self._rgb_render_target
        render_desc.resource_bindings[0].render_target_binding.first_layer = 0
        render_desc.resource_bindings[0].render_target_binding.layer_count = self.env_count
        render_desc.resource_bindings[0].render_target_texture_plane = (
            neo.GpuRenderTargetTexturePlane.Color
        )
        render_desc.resource_bindings[0].access = neo.CustomComputeResourceAccess.ReadOnly
        render_desc.resource_bindings[1].shader_variable_name = "g_ColorObservation"
        render_desc.resource_bindings[1].shared_buffer_handle = self.rgb_observation_buffer
        render_desc.resource_bindings[1].access = neo.CustomComputeResourceAccess.ReadWrite
        render_desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
        render_desc.dispatch.group_count_x = (self.image_width + 7) // 8
        render_desc.dispatch.group_count_y = (self.image_height + 7) // 8
        render_desc.dispatch.group_count_z = self.env_count
        self._rgb_render_pass = self.runtime.create_custom_compute_pass(render_desc)
        if not self._rgb_render_pass.is_valid():
            raise RuntimeError("Failed to create cartpole RGB observation pass.")

    def reset(self, env_ids: "torch.Tensor | list[int] | None" = None) -> "torch.Tensor":
        if env_ids is None:
            env_indices = torch.arange(self.env_count, device=self.action_tensor.device, dtype=torch.int64)
        elif isinstance(env_ids, torch.Tensor):
            env_indices = env_ids.to(device=self.action_tensor.device, dtype=torch.int64)
        else:
            env_indices = torch.tensor(list(env_ids), device=self.action_tensor.device, dtype=torch.int64)

        if env_indices.numel() == 0:
            return self.observation_tensor
        self.reset_mask_tensor.zero_()
        for env_index in env_indices.tolist():
            self.reset_mask_tensor[int(env_index)] = 1

        self.reset_state_tensor.zero_()
        sampled_reset_state = torch.empty(
            (env_indices.numel(), self.OBSERVATION_DIM),
            device=self.reset_state_tensor.device,
            dtype=self.reset_state_tensor.dtype,
        )
        sampled_reset_state[:, 0].uniform_(
            -self.reset_cart_position_range, self.reset_cart_position_range
        )
        sampled_reset_state[:, 1].uniform_(
            -self.reset_cart_velocity_range, self.reset_cart_velocity_range
        )
        sampled_reset_state[:, 2].uniform_(
            -self.reset_pole_angle_range_radians, self.reset_pole_angle_range_radians
        )
        sampled_reset_state[:, 3].uniform_(
            -self.reset_pole_angular_velocity_range, self.reset_pole_angular_velocity_range
        )
        self.reset_state_tensor.index_copy_(0, env_indices, sampled_reset_state)

        if not self.runtime.sync_shared_buffer_from_cuda(self.reset_mask_buffer):
            raise RuntimeError("Failed to synchronize cartpole reset mask from CUDA.")
        if not self.runtime.sync_shared_buffer_from_cuda(self.reset_state_buffer):
            raise RuntimeError("Failed to synchronize cartpole reset state from CUDA.")
        if not self.runtime.execute_custom_compute_pass(self._reset_pass):
            raise RuntimeError("Failed to execute cartpole reset pass.")
        self._sync_outputs_to_cuda()
        self.runtime.end_frame(self._frame)
        self.reset_mask_tensor.zero_()
        if not self.runtime.sync_shared_buffer_from_cuda(self.reset_mask_buffer):
            raise RuntimeError("Failed to clear cartpole reset mask after reset.")
        return self.observation_tensor

    def step(self, action_tensor: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        if list(action_tensor.shape) != [self.env_count]:
            raise ValueError(f"Expected action tensor shape [{self.env_count}], got {list(action_tensor.shape)}.")

        self.action_tensor.copy_(action_tensor.to(device=self.action_tensor.device, dtype=self.action_tensor.dtype))
        if not self.runtime.sync_shared_buffer_from_cuda(self.action_buffer):
            raise RuntimeError("Failed to synchronize cartpole actions from CUDA.")
        if not self.runtime.execute_custom_compute_pass(self._pre_pass):
            raise RuntimeError("Failed to execute cartpole pre-physics control pass.")
        if not self.runtime.step_physics(self._frame):
            raise RuntimeError("Cartpole physics step failed.")
        if not self.runtime.execute_custom_compute_pass(self._post_pass):
            raise RuntimeError("Failed to execute cartpole post-physics observation pass.")
        self._sync_outputs_to_cuda()
        self.runtime.end_frame(self._frame)
        self._frame.frame_index += 1
        self._frame.time_seconds += self._frame.delta_seconds
        return (
            self.observation_tensor,
            self.reward_tensor,
            self.terminated_tensor,
            self.truncated_tensor,
        )

    def render(self) -> "torch.Tensor":
        self.runtime.step_visual_sensors(self._frame)
        if not self.runtime.execute_custom_compute_pass(self._rgb_render_pass):
            raise RuntimeError("Failed to execute cartpole RGB observation pass.")
        if not self.runtime.sync_shared_buffer_to_cuda(self.rgb_observation_buffer):
            raise RuntimeError("Failed to synchronize cartpole RGB observation buffer to CUDA.")
        self.runtime.end_frame(self._frame)
        torch.cuda.synchronize(device=self.rgb_observation_tensor.device)
        return self.rgb_observation_tensor

    def _sync_outputs_to_cuda(self) -> None:
        for handle in (
            self.observation_buffer,
            self.reward_buffer,
            self.terminated_buffer,
            self.truncated_buffer,
            self.episode_steps_buffer,
        ):
            if not self.runtime.sync_shared_buffer_to_cuda(handle):
                raise RuntimeError("Failed to synchronize cartpole outputs back to CUDA.")
        torch.cuda.synchronize(device=self.observation_tensor.device)

    def close(self) -> None:
        if getattr(self, "runtime", None) is None:
            return
        for attr in ("_pre_pass", "_post_pass", "_reset_pass", "_rgb_render_pass"):
            handle = getattr(self, attr, None)
            if handle is not None and handle.is_valid():
                self.runtime.destroy_custom_compute_pass(handle)
        self.runtime.shutdown()
        self.runtime = None


class CartpoleGymnasiumAdapter:
    def __init__(self, env: CartpoleTorchVectorEnv) -> None:
        self.env = env
        if spaces is not None:
            self.single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            self.single_observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(CartpoleTorchVectorEnv.OBSERVATION_DIM,), dtype=np.float32
            )

    def reset(self, env_ids: "torch.Tensor | list[int] | None" = None):
        obs = self.env.reset(env_ids)
        obs_cpu = obs.detach().cpu().numpy()
        info = {}
        return obs_cpu, info

    def step(self, action):
        if np is None:
            raise RuntimeError("NumPy is required for the Gymnasium adapter.")
        action_tensor = torch.as_tensor(action, device=self.env.action_tensor.device, dtype=torch.float32)
        if action_tensor.ndim == 2 and action_tensor.shape == (self.env.env_count, 1):
            action_tensor = action_tensor.squeeze(-1)
        obs, reward, terminated, truncated = self.env.step(action_tensor)
        return (
            obs.detach().cpu().numpy(),
            reward.detach().cpu().numpy(),
            terminated.detach().cpu().numpy().astype(np.bool_),
            truncated.detach().cpu().numpy().astype(np.bool_),
            {},
        )

    def close(self) -> None:
        self.env.close()
