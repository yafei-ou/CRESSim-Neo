from __future__ import annotations

import math
import struct

import cressim_neo as neo
from .torch_env import TorchStagedVectorEnvBase

try:
    import torch
except ImportError as exc:
    raise RuntimeError(
        "cressim_neo_envs.camera_centering_env requires PyTorch to be installed."
    ) from exc


_CAMERA_CENTERING_PRE_VISUAL_SHADER = r"""
#include "structured_buffer_compat.hlsli"

cbuffer CameraCenteringPreVisualConstants
{
    float yawActionScale;
    float pitchActionScale;
    float minPitchRadians;
    float maxPitchRadians;
};

CRESSIM_STRUCTURED_BUFFER(float2, g_Actions);
CRESSIM_STRUCTURED_BUFFER(uint, g_CameraPoseSlots);
CRESSIM_RW_STRUCTURED_BUFFER(float2, g_CameraAngles);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_EntityOrientations);

float4 QuaternionMul(float4 a, float4 b)
{
    return float4(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - dot(a.xyz, b.xyz));
}

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

    float2 angles = CRESSIM_SB_LOAD(g_CameraAngles, envIndex);
    const float2 action = clamp(CRESSIM_SB_LOAD(g_Actions, envIndex), float2(-1.0f, -1.0f),
                                float2(1.0f, 1.0f));
    angles.x += action.x * yawActionScale;
    angles.y = clamp(angles.y + action.y * pitchActionScale, minPitchRadians, maxPitchRadians);
    CRESSIM_SB_STORE(g_CameraAngles, envIndex, angles);

    float yawSin = 0.0f;
    float yawCos = 1.0f;
    sincos(0.5f * angles.x, yawSin, yawCos);
    float pitchSin = 0.0f;
    float pitchCos = 1.0f;
    sincos(0.5f * angles.y, pitchSin, pitchCos);

    const float4 yawQuat = float4(0.0f, yawSin, 0.0f, yawCos);
    const float4 pitchQuat = float4(pitchSin, 0.0f, 0.0f, pitchCos);
    const float4 orientation = QuaternionMul(yawQuat, pitchQuat);

    const uint poseSlot = CRESSIM_SB_LOAD(g_CameraPoseSlots, envIndex);
    CRESSIM_SB_STORE(g_EntityOrientations, poseSlot, orientation);
}
"""


_CAMERA_CENTERING_REWARD_SHADER = r"""
#include "structured_buffer_compat.hlsli"

cbuffer CameraCenteringRewardConstants
{
    float4 rewardParams0;
    float4 rewardParams1;
};

CRESSIM_STRUCTURED_BUFFER(uint, g_CameraPoseSlots);
CRESSIM_STRUCTURED_BUFFER(uint, g_TargetPoseSlots);
CRESSIM_STRUCTURED_BUFFER(float4, g_EntityPositions);
CRESSIM_STRUCTURED_BUFFER(float4, g_EntityOrientations);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_PreviousCenterDistances);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_Rewards);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_Terminated);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_Truncated);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_EpisodeSteps);

float3 QuaternionRotate(float4 q, float3 v)
{
    const float3 t = 2.0f * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

float ComputeCenterDistance(uint envIndex)
{
    const uint cameraPoseSlot = CRESSIM_SB_LOAD(g_CameraPoseSlots, envIndex);
    const uint targetPoseSlot = CRESSIM_SB_LOAD(g_TargetPoseSlots, envIndex);
    const float3 cameraPosition = CRESSIM_SB_LOAD(g_EntityPositions, cameraPoseSlot).xyz;
    const float4 cameraOrientation = CRESSIM_SB_LOAD(g_EntityOrientations, cameraPoseSlot);
    const float3 targetPosition = CRESSIM_SB_LOAD(g_EntityPositions, targetPoseSlot).xyz;
    const float3 worldDelta = targetPosition - cameraPosition;
    const float4 inverseCameraOrientation =
        float4(-cameraOrientation.x, -cameraOrientation.y, -cameraOrientation.z, cameraOrientation.w);
    const float3 viewDelta = QuaternionRotate(inverseCameraOrientation, worldDelta);
    if (viewDelta.z <= 1.0e-4f)
    {
        return rewardParams1.x;
    }

    const float tanHalfVerticalFov = rewardParams1.z;
    const float aspectRatio = rewardParams1.w;
    const float tanHalfHorizontalFov = tanHalfVerticalFov * max(aspectRatio, 1.0e-4f);
    const float2 centered =
        float2(viewDelta.x / (viewDelta.z * tanHalfHorizontalFov),
               -viewDelta.y / (viewDelta.z * tanHalfVerticalFov));
    return length(centered);
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint envIndex = dispatchThreadID.x;
    uint envCount = 0u;
    uint stride = 0u;
    g_CameraPoseSlots.GetDimensions(envCount, stride);
    if (envIndex >= envCount)
    {
        return;
    }

    const uint nextEpisodeStep = CRESSIM_SB_LOAD(g_EpisodeSteps, envIndex) + 1u;
    uint terminated = 0u;
    uint truncated = nextEpisodeStep >= uint(rewardParams1.y + 0.5f) ? 1u : 0u;
    const float previousDistance = CRESSIM_SB_LOAD(g_PreviousCenterDistances, envIndex);
    const float currentDistance = ComputeCenterDistance(envIndex);
    const float progressReward = (previousDistance - currentDistance) * rewardParams0.w;
    float reward = progressReward - rewardParams0.y;
    if (currentDistance <= rewardParams0.x)
    {
        terminated = 1u;
        reward += rewardParams0.z;
    }

    CRESSIM_SB_STORE(g_PreviousCenterDistances, envIndex, currentDistance);
    CRESSIM_SB_STORE(g_Rewards, envIndex, reward);
    CRESSIM_SB_STORE(g_Terminated, envIndex, terminated);
    CRESSIM_SB_STORE(g_Truncated, envIndex, truncated);
    CRESSIM_SB_STORE(g_EpisodeSteps, envIndex, nextEpisodeStep);
}
"""


_CAMERA_CENTERING_RGB_SHADER = r"""
#include "structured_buffer_compat.hlsli"

Texture2DArray<float4> g_ColorTarget;
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_ColorObservation);

float toneMapReinhard(float value)
{
    return value / (1.0 + value);
}

float linearToSrgb(float value)
{
    if (value <= 0.0031308)
    {
        return value * 12.92;
    }
    return 1.055 * pow(abs(value), 1.0 / 2.4) - 0.055;
}

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
    float4 color = g_ColorTarget.Load(int4(int(x), int(y), int(envIndex), 0));
    color.rgb = max(color.rgb, 0.0);
    color.r = linearToSrgb(toneMapReinhard(color.r));
    color.g = linearToSrgb(toneMapReinhard(color.g));
    color.b = linearToSrgb(toneMapReinhard(color.b));
    color = saturate(color);
    CRESSIM_SB_STORE(g_ColorObservation, pixelIndex, color);
}
"""


_CAMERA_CENTERING_RESET_SHADER = r"""
#include "structured_buffer_compat.hlsli"

CRESSIM_STRUCTURED_BUFFER(uint, g_ResetMask);
CRESSIM_STRUCTURED_BUFFER(uint, g_CameraPoseSlots);
CRESSIM_STRUCTURED_BUFFER(uint, g_TargetPoseSlots);
CRESSIM_STRUCTURED_BUFFER(float2, g_BaseCameraAngles);
CRESSIM_STRUCTURED_BUFFER(float4, g_EntityPositions);
CRESSIM_RW_STRUCTURED_BUFFER(float2, g_CameraAngles);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_EntityOrientations);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_PreviousCenterDistances);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_Rewards);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_Terminated);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_Truncated);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_EpisodeSteps);

cbuffer CameraCenteringResetConstants
{
    float4 resetParams;
};

float4 QuaternionMul(float4 a, float4 b)
{
    return float4(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - dot(a.xyz, b.xyz));
}

float3 QuaternionRotate(float4 q, float3 v)
{
    const float3 t = 2.0f * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

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

    const float2 angles = CRESSIM_SB_LOAD(g_BaseCameraAngles, envIndex);
    CRESSIM_SB_STORE(g_CameraAngles, envIndex, angles);

    float yawSin = 0.0f;
    float yawCos = 1.0f;
    sincos(0.5f * angles.x, yawSin, yawCos);
    float pitchSin = 0.0f;
    float pitchCos = 1.0f;
    sincos(0.5f * angles.y, pitchSin, pitchCos);

    const float4 yawQuat = float4(0.0f, yawSin, 0.0f, yawCos);
    const float4 pitchQuat = float4(pitchSin, 0.0f, 0.0f, pitchCos);
    const float4 orientation = QuaternionMul(yawQuat, pitchQuat);

    const uint poseSlot = CRESSIM_SB_LOAD(g_CameraPoseSlots, envIndex);
    CRESSIM_SB_STORE(g_EntityOrientations, poseSlot, orientation);
    const uint targetPoseSlot = CRESSIM_SB_LOAD(g_TargetPoseSlots, envIndex);
    const float3 cameraPosition = CRESSIM_SB_LOAD(g_EntityPositions, poseSlot).xyz;
    const float3 targetPosition = CRESSIM_SB_LOAD(g_EntityPositions, targetPoseSlot).xyz;
    const float3 worldDelta = targetPosition - cameraPosition;
    const float4 inverseCameraOrientation = float4(-orientation.x, -orientation.y, -orientation.z, orientation.w);
    const float3 viewDelta = QuaternionRotate(inverseCameraOrientation, worldDelta);
    float centerDistance = resetParams.x;
    if (viewDelta.z > 1.0e-4f)
    {
        const float tanHalfHorizontalFov = resetParams.z * max(resetParams.w, 1.0e-4f);
        const float2 centered =
            float2(viewDelta.x / (viewDelta.z * tanHalfHorizontalFov),
                   -viewDelta.y / (viewDelta.z * resetParams.z));
        centerDistance = length(centered);
    }
    CRESSIM_SB_STORE(g_PreviousCenterDistances, envIndex, centerDistance);
    CRESSIM_SB_STORE(g_Rewards, envIndex, 0.0f);
    CRESSIM_SB_STORE(g_Terminated, envIndex, 0u);
    CRESSIM_SB_STORE(g_Truncated, envIndex, 0u);
    CRESSIM_SB_STORE(g_EpisodeSteps, envIndex, 0u);
}
"""


def _quat_from_yaw_pitch(yaw_radians: float, pitch_radians: float) -> neo.Quaternion:
    yaw_half = 0.5 * yaw_radians
    pitch_half = 0.5 * pitch_radians
    yaw_sin = math.sin(yaw_half)
    yaw_cos = math.cos(yaw_half)
    pitch_sin = math.sin(pitch_half)
    pitch_cos = math.cos(pitch_half)
    quat = neo.Quaternion()
    quat.x = pitch_sin * yaw_cos
    quat.y = yaw_sin * pitch_cos
    quat.z = -yaw_sin * pitch_sin
    quat.w = yaw_cos * pitch_cos
    return quat


class CameraCenteringTorchVectorEnv(TorchStagedVectorEnvBase):
    ACTION_DIM = 2

    def __init__(
        self,
        env_count: int = 32,
        max_episode_steps: int = 120,
        action_scale_yaw_degrees: float = 0.2,
        action_scale_pitch_degrees: float = 0.2,
        base_yaw_degrees: float = 0.0,
        base_pitch_degrees: float = -12.5,
        reset_yaw_min_degrees: float = -16.0,
        reset_yaw_max_degrees: float = 16.0,
        reset_pitch_min_degrees: float = -6.5,
        reset_pitch_max_degrees: float = 30.0,
        min_pitch_degrees: float = -35.0,
        max_pitch_degrees: float = 35.0,
        success_center_threshold: float = 0.06,
        image_width: int = 128,
        image_height: int = 128,
    ) -> None:
        super().__init__(env_count)
        self.max_episode_steps = max_episode_steps
        self.action_scale_yaw = math.radians(action_scale_yaw_degrees)
        self.action_scale_pitch = math.radians(action_scale_pitch_degrees)
        self.base_yaw = math.radians(base_yaw_degrees)
        self.base_pitch = math.radians(base_pitch_degrees)
        self.min_pitch = math.radians(min_pitch_degrees)
        self.max_pitch = math.radians(max_pitch_degrees)
        self.reset_yaw_min = math.radians(reset_yaw_min_degrees)
        self.reset_yaw_max = math.radians(reset_yaw_max_degrees)
        self.reset_pitch_min = math.radians(reset_pitch_min_degrees)
        self.reset_pitch_max = math.radians(reset_pitch_max_degrees)
        if self.reset_yaw_min > self.reset_yaw_max:
            raise ValueError("reset_yaw_min_degrees must be <= reset_yaw_max_degrees.")
        if self.reset_pitch_min > self.reset_pitch_max:
            raise ValueError("reset_pitch_min_degrees must be <= reset_pitch_max_degrees.")
        self.reset_pitch_min = max(self.reset_pitch_min, self.min_pitch)
        self.reset_pitch_max = min(self.reset_pitch_max, self.max_pitch)
        if self.reset_pitch_min > self.reset_pitch_max:
            raise ValueError("Reset pitch range does not overlap the allowed pitch limits.")
        self.success_center_threshold = success_center_threshold
        self.image_width = image_width
        self.image_height = image_height

        config = neo.RuntimeConfig()
        config.gpu_device_desc.preferred_backend = neo.GpuBackend.Vulkan
        config.gpu_device_desc.enable_validation = False
        config.physics_desc.enable_blocking_readback = False
        config.scene_layout.env_count = env_count
        config.scene_layout.max_renderable_objects_per_env = 4
        config.scene_layout.max_lights_per_env = 1
        config.scene_layout.max_cameras_per_env = 1

        self.runtime = neo.Runtime()
        if not self.runtime.initialize(config):
            raise RuntimeError("Failed to initialize camera-centering runtime.")

        self._initialize_render_targets()
        self._camera_pose_slots: list[int] = []
        self._target_pose_slots: list[int] = []
        self._author_scene(self.runtime.world())
        self.runtime.prepare()

        if not self.runtime.upload_world():
            self.close()
            raise RuntimeError("Failed to upload prepared camera-centering world.")

        self._create_shared_buffers()
        self._populate_lookup_buffers()
        self._create_custom_passes()
        self._end_frame(self.runtime, advance=False)

    def _initialize_render_targets(self) -> None:
        color_desc = neo.GpuRenderTargetDesc()
        color_desc.width = self.image_width
        color_desc.height = self.image_height
        color_desc.array_size = self.env_count
        color_desc.color = True
        color_desc.depth = True
        color_desc.color_format = neo.TextureFormat.RGBA16Float
        color_desc.layered_rendering = True
        color_desc.shader_readable = True
        color_desc.debug_name = "CameraCentering.ColorTarget"
        self._color_render_target = self.runtime.create_render_target(color_desc)
        if not self.runtime.is_valid_render_target(self._color_render_target):
            raise RuntimeError("Failed to create camera-centering RGB render target.")

        resources = self.runtime.resources()
        self._target_mesh = resources.register_mesh(
            neo.make_cube_mesh(0.20, "CameraCentering.TargetMesh")
        )
        self._ground_mesh = resources.register_mesh(
            neo.make_box_mesh(neo.Float3(2.2, 0.05, 2.2), "CameraCentering.GroundMesh")
        )

    def _make_material(
        self, debug_name: str, base_color: neo.Float3, roughness: float
    ) -> neo.MaterialHandle:
        material_desc = neo.MaterialResourceDesc()
        material_desc.debug_name = debug_name
        material_desc.base_color = base_color
        material_desc.metallic = 0.0
        material_desc.roughness = roughness
        return self.runtime.resources().register_material(material_desc)

    def _author_scene(self, world: neo.World) -> None:
        ground_material = self._make_material(
            "CameraCentering.GroundMaterial", neo.Float3(0.56, 0.57, 0.60), 0.95
        )
        camera_orientation = _quat_from_yaw_pitch(self.base_yaw, self.base_pitch)
        for env_index in range(self.env_count):
            z_offset = float(env_index) * 4.0

            ground_entity = world.create_entity(env_index)
            ground_transform = neo.TransformComponent()
            ground_transform.world_transform.position = neo.Float3(0.0, -0.05, z_offset)
            world.set_transform(ground_entity, ground_transform)
            ground_renderer = neo.MeshRendererComponent()
            ground_renderer.mesh = self._ground_mesh
            ground_renderer.material = ground_material
            ground_renderer.segmentation_id = 2000 + env_index
            ground_renderer.visible = True
            world.set_mesh_renderer(ground_entity, ground_renderer)

            target_entity = world.create_entity(env_index)
            target_transform = neo.TransformComponent()
            target_transform.world_transform.position = neo.Float3(0.0, 0.20, z_offset + 0.8)
            world.set_transform(target_entity, target_transform)
            target_renderer = neo.MeshRendererComponent()
            target_renderer.mesh = self._target_mesh
            target_renderer.material = self._make_material(
                f"CameraCentering.TargetMaterial.{env_index}",
                neo.Float3(0.90, 0.28, 0.18),
                0.35,
            )
            target_renderer.segmentation_id = 1000 + env_index
            target_renderer.visible = True
            world.set_mesh_renderer(target_entity, target_renderer)
            self._target_pose_slots.append(world.entity_pose_slot(target_entity))

            light_entity = world.create_entity(env_index)
            light = neo.DirectionalLightComponent()
            light.direction = neo.Float3(-0.35, -1.0, 0.25)
            light.color = neo.Float3(1.0, 1.0, 1.0)
            light.intensity = 7.0
            light.casts_shadows = False
            world.set_directional_light(light_entity, light)

            camera_position = neo.Float3(0.0, 1.0, z_offset - 2.8)

            color_camera_entity = world.create_entity(env_index)
            color_transform = neo.TransformComponent()
            color_transform.world_transform.position = camera_position
            color_transform.world_transform.rotation = camera_orientation
            world.set_transform(color_camera_entity, color_transform)
            color_camera = neo.CameraComponent()
            color_camera.product = neo.CameraProduct.ColorDepth
            color_camera.vertical_fov_degrees = 50.0
            color_camera.output.mode = neo.RenderOutputMode.ExplicitSurface
            color_camera.output.binding = neo.GpuRenderTargetBinding()
            color_camera.output.binding.target = self._color_render_target
            color_camera.output.binding.first_layer = env_index
            color_camera.output.binding.layer_count = 1
            color_camera.output_width = self.image_width
            color_camera.output_height = self.image_height
            color_camera.clear_color = True
            color_camera.clear_depth = True
            color_camera.clear_color_value = neo.Float4(0.03, 0.04, 0.06, 1.0)
            world.set_camera(color_camera_entity, color_camera)

            self._camera_pose_slots.append(world.entity_pose_slot(color_camera_entity))

    def _create_shared_buffers(self) -> None:
        self.action_buffer, self.action_tensor = self._register_shared_buffer(
            self.runtime,
            "CameraCentering.Actions",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.Float,
            element_stride_bytes=8,
            shape=[self.env_count, self.ACTION_DIM],
        )
        self.rgb_observation_buffer, self.rgb_observation_tensor = self._register_shared_buffer(
            self.runtime,
            "CameraCentering.RgbObservation",
            self.env_count * self.image_width * self.image_height,
            neo.SharedBufferTensorDTypeCode.Float,
            element_stride_bytes=16,
            shape=[self.env_count, self.image_height, self.image_width, 4],
        )
        self.reward_buffer, self.reward_tensor = self._register_shared_buffer(
            self.runtime,
            "CameraCentering.Rewards",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.Float,
        )
        self.previous_center_distances_buffer, self.previous_center_distances_tensor = self._register_shared_buffer(
            self.runtime,
            "CameraCentering.PreviousCenterDistances",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.Float,
        )
        self.terminated_buffer, self.terminated_tensor = self._register_shared_buffer(
            self.runtime,
            "CameraCentering.Terminated",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.UInt,
        )
        self.truncated_buffer, self.truncated_tensor = self._register_shared_buffer(
            self.runtime,
            "CameraCentering.Truncated",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.UInt,
        )
        self.episode_steps_buffer, self.episode_steps_tensor = self._register_shared_buffer(
            self.runtime,
            "CameraCentering.EpisodeSteps",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.UInt,
        )
        self.reset_mask_buffer, self.reset_mask_tensor = self._register_shared_buffer(
            self.runtime,
            "CameraCentering.ResetMask",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.UInt,
        )
        self.camera_pose_slots_buffer, self.camera_pose_slots_tensor = self._register_shared_buffer(
            self.runtime,
            "CameraCentering.CameraPoseSlots",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.UInt,
            shape=[self.env_count],
        )
        self.target_pose_slots_buffer, self.target_pose_slots_tensor = self._register_shared_buffer(
            self.runtime,
            "CameraCentering.TargetPoseSlots",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.UInt,
            shape=[self.env_count],
        )
        self.base_camera_angles_buffer, self.base_camera_angles_tensor = self._register_shared_buffer(
            self.runtime,
            "CameraCentering.BaseCameraAngles",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.Float,
            element_stride_bytes=8,
            shape=[self.env_count, 2],
        )
        self.camera_angles_buffer, self.camera_angles_tensor = self._register_shared_buffer(
            self.runtime,
            "CameraCentering.CameraAngles",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.Float,
            element_stride_bytes=8,
            shape=[self.env_count, 2],
        )
    def _populate_lookup_buffers(self) -> None:
        device = self.action_tensor.device
        base_angles = [(self.base_yaw, self.base_pitch) for _ in range(self.env_count)]
        self.action_tensor.zero_()
        self.rgb_observation_tensor.zero_()
        self.reward_tensor.zero_()
        self.previous_center_distances_tensor.zero_()
        self.terminated_tensor.zero_()
        self.truncated_tensor.zero_()
        self.episode_steps_tensor.zero_()
        self.reset_mask_tensor.zero_()
        self.camera_pose_slots_tensor.copy_(
            torch.tensor(
                self._camera_pose_slots,
                device=device,
                dtype=self.camera_pose_slots_tensor.dtype,
            )
        )
        self.target_pose_slots_tensor.copy_(
            torch.tensor(
                self._target_pose_slots,
                device=device,
                dtype=self.target_pose_slots_tensor.dtype,
            )
        )
        self.base_camera_angles_tensor.copy_(
            torch.tensor(
                base_angles,
                device=device,
                dtype=self.base_camera_angles_tensor.dtype,
            )
        )
        self.camera_angles_tensor.copy_(
            torch.tensor(
                base_angles,
                device=device,
                dtype=self.camera_angles_tensor.dtype,
            )
        )
        self._sync_from_cuda(
            self.runtime,
            [
                self.action_buffer,
                self.rgb_observation_buffer,
                self.reward_buffer,
                self.previous_center_distances_buffer,
                self.terminated_buffer,
                self.truncated_buffer,
                self.episode_steps_buffer,
                self.reset_mask_buffer,
                self.camera_pose_slots_buffer,
                self.target_pose_slots_buffer,
                self.base_camera_angles_buffer,
                self.camera_angles_buffer,
            ],
        )

    def _create_custom_passes(self) -> None:
        def bind(
            desc: neo.CustomComputePassDesc,
            specs: list[tuple[str, object, str, object]],
        ) -> None:
            desc.resource_bindings = [
                neo.CustomComputeResourceBindingDesc() for _ in range(len(specs))
            ]
            for binding, (name, handle, key, access) in zip(desc.resource_bindings, specs):
                binding.shader_variable_name = name
                binding.access = access
                if handle is not None:
                    binding.shared_buffer_handle = handle
                else:
                    binding.resource_key = key

        pre_desc = neo.CustomComputePassDesc()
        pre_desc.debug_name = "CameraCentering.PreVisualControl"
        pre_desc.shader_source = _CAMERA_CENTERING_PRE_VISUAL_SHADER
        pre_desc.thread_group_size_x = 64
        bind(
            pre_desc,
            [
                ("g_Actions", self.action_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_CameraPoseSlots", self.camera_pose_slots_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_CameraAngles", self.camera_angles_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_EntityOrientations", None, "entity.orientations", neo.CustomComputeResourceAccess.ReadWrite),
            ],
        )
        pre_desc.constant_buffer_variable_name = "CameraCenteringPreVisualConstants"
        pre_desc.constant_buffer_size_bytes = 16
        pre_desc.constant_data = list(
            struct.pack(
                "<4f",
                self.action_scale_yaw,
                self.action_scale_pitch,
                self.min_pitch,
                self.max_pitch,
            )
        )
        pre_desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
        pre_desc.dispatch.group_count_x = (self.env_count + 63) // 64
        self._pre_pass = self._register_custom_pass(self.runtime, pre_desc)

        reward_desc = neo.CustomComputePassDesc()
        reward_desc.debug_name = "CameraCentering.PostVisualReward"
        reward_desc.shader_source = _CAMERA_CENTERING_REWARD_SHADER
        reward_desc.thread_group_size_x = 64
        reward_desc.resource_bindings = [neo.CustomComputeResourceBindingDesc() for _ in range(9)]
        reward_specs = [
            ("g_CameraPoseSlots", self.camera_pose_slots_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_TargetPoseSlots", self.target_pose_slots_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_EntityPositions", None, "entity.positions", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_EntityOrientations", None, "entity.orientations", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_PreviousCenterDistances", self.previous_center_distances_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
            ("g_Rewards", self.reward_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
            ("g_Terminated", self.terminated_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
            ("g_Truncated", self.truncated_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
            ("g_EpisodeSteps", self.episode_steps_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
        ]
        for binding, (name, handle, key, access) in zip(reward_desc.resource_bindings, reward_specs):
            binding.shader_variable_name = name
            binding.access = access
            if handle is not None:
                binding.shared_buffer_handle = handle
            else:
                binding.resource_key = key
        reward_desc.constant_buffer_variable_name = "CameraCenteringRewardConstants"
        reward_desc.constant_buffer_size_bytes = 32
        step_penalty = 0.01
        success_bonus = 1.0
        progress_scale = 1.0
        offscreen_distance = 2.0
        tan_half_vertical_fov = math.tan(math.radians(25.0))
        aspect_ratio = float(self.image_width) / max(float(self.image_height), 1.0)
        reward_desc.constant_data = list(
            struct.pack(
                "<8f",
                self.success_center_threshold,
                step_penalty,
                success_bonus,
                progress_scale,
                offscreen_distance,
                float(self.max_episode_steps),
                tan_half_vertical_fov,
                aspect_ratio,
            )
        )
        reward_desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
        reward_desc.dispatch.group_count_x = (self.env_count + 63) // 64
        self._reward_pass = self._register_custom_pass(self.runtime, reward_desc)

        rgb_desc = neo.CustomComputePassDesc()
        rgb_desc.debug_name = "CameraCentering.RgbObservation"
        rgb_desc.shader_source = _CAMERA_CENTERING_RGB_SHADER
        rgb_desc.thread_group_size_x = 8
        rgb_desc.thread_group_size_y = 8
        rgb_desc.resource_bindings = [neo.CustomComputeResourceBindingDesc() for _ in range(2)]
        rgb_desc.resource_bindings[0].shader_variable_name = "g_ColorTarget"
        rgb_desc.resource_bindings[0].render_target_binding = neo.GpuRenderTargetBinding()
        rgb_desc.resource_bindings[0].render_target_binding.target = self._color_render_target
        rgb_desc.resource_bindings[0].render_target_binding.first_layer = 0
        rgb_desc.resource_bindings[0].render_target_binding.layer_count = self.env_count
        rgb_desc.resource_bindings[0].render_target_texture_plane = neo.GpuRenderTargetTexturePlane.Color
        rgb_desc.resource_bindings[0].access = neo.CustomComputeResourceAccess.ReadOnly
        rgb_desc.resource_bindings[1].shader_variable_name = "g_ColorObservation"
        rgb_desc.resource_bindings[1].shared_buffer_handle = self.rgb_observation_buffer
        rgb_desc.resource_bindings[1].access = neo.CustomComputeResourceAccess.ReadWrite
        rgb_desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
        rgb_desc.dispatch.group_count_x = (self.image_width + 7) // 8
        rgb_desc.dispatch.group_count_y = (self.image_height + 7) // 8
        rgb_desc.dispatch.group_count_z = self.env_count
        self._rgb_pass = self._register_custom_pass(self.runtime, rgb_desc)

        reset_desc = neo.CustomComputePassDesc()
        reset_desc.debug_name = "CameraCentering.Reset"
        reset_desc.shader_source = _CAMERA_CENTERING_RESET_SHADER
        reset_desc.thread_group_size_x = 64
        bind(
            reset_desc,
            [
                ("g_ResetMask", self.reset_mask_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_CameraPoseSlots", self.camera_pose_slots_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_TargetPoseSlots", self.target_pose_slots_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_BaseCameraAngles", self.base_camera_angles_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_EntityPositions", None, "entity.positions", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_CameraAngles", self.camera_angles_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_EntityOrientations", None, "entity.orientations", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_PreviousCenterDistances", self.previous_center_distances_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_Rewards", self.reward_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_Terminated", self.terminated_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_Truncated", self.truncated_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_EpisodeSteps", self.episode_steps_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
            ],
        )
        reset_desc.constant_buffer_variable_name = "CameraCenteringResetConstants"
        reset_desc.constant_buffer_size_bytes = 16
        reset_desc.constant_data = list(
            struct.pack(
                "<4f",
                offscreen_distance,
                0.0,
                tan_half_vertical_fov,
                aspect_ratio,
            )
        )
        reset_desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
        reset_desc.dispatch.group_count_x = (self.env_count + 63) // 64
        self._reset_pass = self._register_custom_pass(self.runtime, reset_desc)

    def _sync_step_outputs_to_cuda(self) -> None:
        self._sync_to_cuda(
            self.runtime,
            [
                self.rgb_observation_buffer,
                self.reward_buffer,
                self.terminated_buffer,
                self.truncated_buffer,
                self.episode_steps_buffer,
            ],
            device=self.rgb_observation_tensor.device,
        )

    def _sample_reset_camera_angles(self, env_indices: "torch.Tensor") -> None:
        sampled_angles = torch.empty(
            (env_indices.numel(), 2),
            device=self.base_camera_angles_tensor.device,
            dtype=self.base_camera_angles_tensor.dtype,
        )
        sampled_angles[:, 0].uniform_(self.reset_yaw_min, self.reset_yaw_max)
        sampled_angles[:, 1].uniform_(self.reset_pitch_min, self.reset_pitch_max)
        self.base_camera_angles_tensor[env_indices] = sampled_angles

    def _render_rgb_observation(self) -> None:
        self.runtime.step_visual_sensors(self._frame)
        if not self.runtime.execute_custom_compute_pass(self._rgb_pass):
            raise RuntimeError("Failed to execute camera-centering RGB pass.")

    def reset(self, env_ids: "torch.Tensor | list[int] | None" = None) -> "torch.Tensor":
        if env_ids is None:
            env_indices = torch.arange(
                self.env_count, device=self.action_tensor.device, dtype=torch.int64
            )
        elif isinstance(env_ids, torch.Tensor):
            env_indices = env_ids.to(device=self.action_tensor.device, dtype=torch.int64)
        else:
            env_indices = torch.tensor(
                list(env_ids), device=self.action_tensor.device, dtype=torch.int64
            )
        if env_indices.numel() == 0:
            return self.rgb_observation_tensor

        self._sample_reset_camera_angles(env_indices)
        self.reset_mask_tensor.zero_()
        for env_index in env_indices.tolist():
            self.reset_mask_tensor[int(env_index)] = 1
        self.action_tensor.zero_()
        self._sync_from_cuda(
            self.runtime,
            [
                self.reset_mask_buffer,
                self.action_buffer,
                self.base_camera_angles_buffer,
            ],
        )
        if not self.runtime.execute_custom_compute_pass(self._reset_pass):
            raise RuntimeError("Failed to execute camera-centering reset pass.")
        self._render_rgb_observation()
        self._sync_to_cuda(
            self.runtime,
            [
                self.rgb_observation_buffer,
                self.reward_buffer,
                self.terminated_buffer,
                self.truncated_buffer,
                self.episode_steps_buffer,
            ],
            device=self.rgb_observation_tensor.device,
        )
        self._end_frame(self.runtime, advance=False)
        return self.rgb_observation_tensor

    def step(
        self, action_tensor: "torch.Tensor"
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        if list(action_tensor.shape) != [self.env_count, self.ACTION_DIM]:
            raise ValueError(
                f"Expected action tensor shape [{self.env_count}, {self.ACTION_DIM}], "
                f"got {list(action_tensor.shape)}."
            )
        self.action_tensor.copy_(
            action_tensor.to(device=self.action_tensor.device, dtype=self.action_tensor.dtype)
        )
        self._sync_from_cuda(self.runtime, [self.action_buffer])
        if not self.runtime.execute_custom_compute_pass(self._pre_pass):
            raise RuntimeError("Failed to execute camera-centering pre-visual pass.")
        if not self.runtime.execute_custom_compute_pass(self._reward_pass):
            raise RuntimeError("Failed to execute camera-centering reward pass.")
        self.runtime.step_visual_sensors(self._frame)
        if not self.runtime.execute_custom_compute_pass(self._rgb_pass):
            raise RuntimeError("Failed to execute camera-centering RGB pass.")
        self._sync_step_outputs_to_cuda()
        self._end_frame(self.runtime, advance=True)
        return (
            self.rgb_observation_tensor,
            self.reward_tensor,
            self.terminated_tensor,
            self.truncated_tensor,
        )

    def render(self) -> "torch.Tensor":
        self._render_rgb_observation()
        self._sync_to_cuda(
            self.runtime, [self.rgb_observation_buffer], device=self.rgb_observation_tensor.device
        )
        self.runtime.end_frame(self._frame)
        torch.cuda.synchronize(device=self.rgb_observation_tensor.device)
        return self.rgb_observation_tensor

    def close(self) -> None:
        self.close_runtime(getattr(self, "runtime", None))
        self.runtime = None
