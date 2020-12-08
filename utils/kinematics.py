import numpy as np

from pydrake.all import (
    JacobianWrtVariable, Integrator, LeafSystem, BasicVector, 
    ConstantVectorSource, RigidTransform, SpatialVelocity
)

class InverseKinematics(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self.plant = plant
        self.plant_context = plant.CreateDefaultContext()
        self.iiwa = plant.GetModelInstanceByName("iiwa7")
        self.P = plant.GetBodyByName("base_link").body_frame()
        self.W = plant.world_frame()

        self.DeclareVectorInputPort("paddle_desired_velocity", BasicVector(6))
        # self.DeclareVectorInputPort("paddle_desired_angular_velocity", BasicVector(3))

        self.DeclareVectorInputPort("iiwa_pos_measured", BasicVector(7))
        self.DeclareVectorOutputPort("iiwa_velocity", BasicVector(7), self.CalcOutput)
        self.iiwa_start = plant.GetJointByName("iiwa_joint_1").velocity_start()
        self.iiwa_end = plant.GetJointByName("iiwa_joint_7").velocity_start()

    def CalcOutput(self, context, output):
        q = self.GetInputPort("iiwa_pos_measured").Eval(context)
        V_P_desired = self.GetInputPort("paddle_desired_velocity").Eval(context)
        # pos_P_desired = self.GetInputPort("paddle_desired_velocity").Eval(context)
        self.plant.SetPositions(self.plant_context, self.iiwa, q)
        J_P = self.plant.CalcJacobianSpatialVelocity(
            self.plant_context, JacobianWrtVariable.kV, 
            self.P, [0,0,0], self.W, self.W)
        J_P = J_P[:,self.iiwa_start:self.iiwa_end+1]

        v = np.linalg.pinv(J_P).dot(V_P_desired)
        # #overwrite for debugging
        # v = [np.pi/4,0,0,0,0,0,0]
        output.SetFromVector(v)


class VelocityMirror(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self.plant = plant
        self.plant_context = plant.CreateDefaultContext()
        self.Ball = plant.GetBodyByName("ball")
        self.W = plant.world_frame()

        self.DeclareVectorInputPort("ball_pose", BasicVector(3))
        self.DeclareVectorInputPort("ball_velocity", BasicVector(6))
        self.DeclareVectorOutputPort("mirror_velocity", BasicVector(6), self.CalcOutput)
        self.iiwa_start = plant.GetJointByName("iiwa_joint_1").velocity_start()
        self.iiwa_end = plant.GetJointByName("iiwa_joint_7").velocity_start()

    def CalcOutput(self, context, output):
        X_B = RigidTransform(self.GetInputPort("ball_pose").Eval(context))
        V_B = SpatialVelocity(self.GetInputPort("ball_velocity").Eval(context))
        self.plant.SetFreeBodyPose(self.plant_context, self.Ball, X_B)
        self.plant.SetFreeBodySpatialVelocity(self.Ball, V_B, self.plant_context)
        v_Ball = self.plant.EvalBodySpatialVelocityInWorld(self.plant_context, self.Ball).translational()
        # print(v_Ball)
        output.SetFromVector(np.array([0, 0, 0, v_Ball[0], v_Ball[1], -v_Ball[2]]))




class PaddleTilt(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self.plant = plant
        self.plant_context = plant.CreateDefaultContext()
        self.iiwa = plant.GetModelInstanceByName("iiwa7")

        self.DeclareVectorInputPort("assigned_q", BasicVector(7))
        self.DeclareVectorInputPort("iiwa_pos_measured", BasicVector(7))
        self.DeclareVectorOutputPort("updated_q", BasicVector(7), self.CalcOutput)

    def CalcOutput(self, context, output):
        current_q = self.GetInputPort("iiwa_pos_measured").Eval(context)
        assigned_q = self.GetInputPort("assigned_q").Eval(context)
        self.plant.SetPositions(self.plant_context, self.iiwa, current_q)
        P_Paddle = self.plant.EvalBodyPoseInWorld(self.plant_context, self.plant.GetBodyByName("base_link")).translation()
        P_x, P_y = P_Paddle[0], P_Paddle[1]
        theta_X, theta_Y = 0, 0
        if abs(P_x-0.88) > 0.2: 
            theta_X = np.arctan(.1*(P_x-0.88)**3)
        if abs(P_y) > 0.2:
            theta_Y = np.arctan(.1*(P_y)**3)
        # print(f"Previous assignment: {assigned_q[5]}, {assigned_q[6]}")
        assigned_q[5] = assigned_q[5] - theta_X
        assigned_q[6] = assigned_q[6] + theta_Y
        # print(f"New assignment: {assigned_q[5]}, {assigned_q[6]}\n")
        # #overwrite for debugging
        # v = [np.pi/4,0,0,0,0,0,0]
        output.SetFromVector(assigned_q)