#ifndef CRESSIM_NEO_PHYSICS_SOLVER_CONFIG_HLSLI
#define CRESSIM_NEO_PHYSICS_SOLVER_CONFIG_HLSLI

// Keep this order in lockstep with GpuPhysicsSolverConfig.
cbuffer PhysicsSolverConfigBuffer
{
    float4 g_ContactSolverConfig0;
    float4 g_ContactSolverConfig1;
    float4 g_RigidSolverConfig0;
    float4 g_RigidSolverConfig1;
    float4 g_JointSolverConfig0;
    float4 g_JointSolverConfig1;
    float4 g_SoftSolverConfig0;
    float4 g_SoftSolverConfig1;
    float4 g_FluidSolverConfig0;
    float4 g_FluidSolverConfig1;
};

#define kContactSlop g_ContactSolverConfig0.x
#define kManifoldMergeDistance (g_ContactSolverConfig0.y * kContactSlop)
#define kSoftContactRelaxation g_ContactSolverConfig0.z
#define kSoftMaxCorrectionPerIter g_ContactSolverConfig0.w
#define kRestitutionVelocityThreshold g_ContactSolverConfig1.x
#define kRestitutionPenetrationThreshold (g_ContactSolverConfig1.y * kContactSlop)
#define kRigidRestitutionThreshold g_ContactSolverConfig1.z
#define kPositionFrictionMinDistance g_ContactSolverConfig1.w
#define kBaumgarte g_RigidSolverConfig0.x
#define kRigidDepenetrationRelaxation g_RigidSolverConfig0.y
#define kRigidMaxDepenetrationCorrectionPerIter g_RigidSolverConfig0.z
#define kRigidVelocityPenetrationStiffness g_RigidSolverConfig0.w
#define kMaxTotalTranslationCorrectionPerIter g_RigidSolverConfig1.x
#define kMaxTotalRotationCorrectionPerIter g_RigidSolverConfig1.y
#define kMaxTotalLinearVelocityCorrectionPerIter g_RigidSolverConfig1.z
#define kMaxTotalAngularVelocityCorrectionPerIter g_RigidSolverConfig1.w
#define kBallJointRelaxation g_JointSolverConfig0.x
#define kJointRelaxation g_JointSolverConfig0.y
#define kMaxJointError g_JointSolverConfig0.z
#define kMaxJointTranslationCorrection g_JointSolverConfig0.w
#define kMaxJointAngularCorrection g_JointSolverConfig1.x
#define kHingeTranslationRegularization g_JointSolverConfig1.y
#define kHingeAngularRegularization g_JointSolverConfig1.z
#define kSliderTranslationRegularization g_JointSolverConfig1.y
#define kSliderAngularRegularization g_JointSolverConfig1.z
#define kJointRegularization g_JointSolverConfig1.y
#define kMinXpbdDt g_JointSolverConfig1.w
#define kSoftInternalRelaxation g_SoftSolverConfig0.x
#define kFluidSolveCoefficient g_FluidSolverConfig0.x
#define kFluidRelaxationFactor g_FluidSolverConfig0.y
#define kFluidBoundaryDensityScale g_FluidSolverConfig0.z
#define kFluidBoundaryDeltaScale g_FluidSolverConfig0.w

#endif // CRESSIM_NEO_PHYSICS_SOLVER_CONFIG_HLSLI
