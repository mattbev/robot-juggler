import numpy as np

from pydrake.all import (
    JacobianWrtVariable, Integrator, LeafSystem, BasicVector, 
    ConstantVectorSource, RigidTransform, SpatialVelocity, RollPitchYaw
)

class InverseKinematics(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self.plant = plant
        self.plant_context = plant.CreateDefaultContext()
        self.iiwa = plant.GetModelInstanceByName("iiwa7")
        self.P = plant.GetBodyByName("base_link").body_frame()
        self.W = plant.world_frame()

        self.DeclareVectorInputPort("paddle_desired_velocity", BasicVector(3))
        self.DeclareVectorInputPort("paddle_desired_angular_velocity", BasicVector(3))
        self.DeclareVectorInputPort("iiwa_pos_measured", BasicVector(7))
        self.DeclareVectorOutputPort("iiwa_velocity", BasicVector(7), self.CalcOutput)
        self.iiwa_start = plant.GetJointByName("iiwa_joint_1").velocity_start()
        self.iiwa_end = plant.GetJointByName("iiwa_joint_7").velocity_start()

    def CalcOutput(self, context, output):
        q = self.GetInputPort("iiwa_pos_measured").Eval(context)
        v_P_desired = self.GetInputPort("paddle_desired_velocity").Eval(context)
        w_P_desired = self.GetInputPort("paddle_desired_angular_velocity").Eval(context)
        # pos_P_desired = self.GetInputPort("paddle_desired_velocity").Eval(context)
        self.plant.SetPositions(self.plant_context, self.iiwa, q)
        J_P = self.plant.CalcJacobianSpatialVelocity(
            self.plant_context, JacobianWrtVariable.kV, 
            self.P, [0,0,0], self.W, self.W)
        J_P = J_P[:,self.iiwa_start:self.iiwa_end+1]
        # print(w_P_desired)
        v = np.linalg.pinv(J_P).dot(np.hstack([w_P_desired, v_P_desired]))
        # v = np.linalg.pinv(J_P).dot(np.hstack([[0, 0, 0], v_P_desired]))
        # v = np.linalg.pinv(J_P).dot(np.hstack([[0, np.pi, 0], [0, 0, 0]]))

        # #overwrite for debugging
        # v = [0,0,0,0,0,0,0]
        output.SetFromVector(v)


class VelocityMirror(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self.plant = plant
        self.plant_context = plant.CreateDefaultContext()
        self.iiwa = plant.GetModelInstanceByName("iiwa7")
        self.Paddle = plant.GetBodyByName("base_link")
        self.W = plant.world_frame()

        self.DeclareVectorInputPort("iiwa_pos_measured", BasicVector(7))
        self.DeclareVectorInputPort("iiwa_velocity_estimated", BasicVector(7))
        self.DeclareVectorInputPort("ball_pose", BasicVector(3))
        self.DeclareVectorInputPort("ball_velocity", BasicVector(3))
        self.DeclareVectorOutputPort("mirror_velocity", BasicVector(3), self.CalcOutput)

    def CalcOutput(self, context, output):
        q = self.GetInputPort("iiwa_pos_measured").Eval(context)
        v = self.GetInputPort("iiwa_velocity_estimated").Eval(context)
        p_Ball_xy = np.array(self.GetInputPort("ball_pose").Eval(context))[:2]
        v_Ball = np.array(self.GetInputPort("ball_velocity").Eval(context))
        v_Ball_xy, v_Ball_z = v_Ball[:2], np.array([v_Ball[2]])
        if v_Ball_z >= 0:
            v_Ball_z *= -1
        else:
            v_Ball_z *= -1   
        self.plant.SetPositionsAndVelocities(self.plant_context, self.iiwa, np.hstack([q, v]))
        p_Paddle_xy = np.array(self.plant.EvalBodyPoseInWorld(self.plant_context, self.Paddle).translation())[:2]
        v_Paddle_xy = np.array(self.plant.EvalBodySpatialVelocityInWorld(self.plant_context, self.Paddle).translational())[:2]
        
        K_p = 4
        K_d = 1
        v_P_desired = K_p*(p_Ball_xy - p_Paddle_xy) + K_d*(v_Ball_xy - v_Paddle_xy)
        v_P_desired = np.concatenate((v_P_desired, v_Ball_z))
        # Tune down with radial shape (1-x^2 - y^2)
        scale = (1 - (p_Ball_xy[0] - .88)**2 - p_Ball_xy[1]**2)
        v_P_desired[2] *= scale
        output.SetFromVector(v_P_desired)


class AngularVelocityTilt(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self.plant = plant
        self.plant_context = plant.CreateDefaultContext()
        self.iiwa = plant.GetModelInstanceByName("iiwa7")
        self.Ball = plant.GetBodyByName("ball")
        self.P = plant.GetBodyByName("base_link")
        self.W = plant.world_frame()

        self.DeclareVectorInputPort("iiwa_pos_measured", BasicVector(7))
        self.DeclareVectorInputPort("ball_pose", BasicVector(3))
        self.DeclareVectorOutputPort("angular_velocity", BasicVector(3), self.CalcOutput)
        self.iiwa_start = plant.GetJointByName("iiwa_joint_1").velocity_start()
        self.iiwa_end = plant.GetJointByName("iiwa_joint_7").velocity_start()

    def CalcOutput(self, context, output):
        q = self.GetInputPort("iiwa_pos_measured").Eval(context)
        X_B = self.GetInputPort("ball_pose").Eval(context)
        self.plant.SetPositions(self.plant_context, self.iiwa, q)
        R_P = RollPitchYaw(self.plant.EvalBodyPoseInWorld(self.plant_context, self.P).rotation()).vector()
        roll_current, pitch_current, yaw_current = R_P[0], R_P[1], R_P[2]
        # B_x, B_y = X_B[0], X_B[1]
        
        k = np.array([1, 4])
        centerpoint = np.array([0.9, 0])
        # roll_des = np.sign(B_y) * np.arctan(k * (1 - np.cos(B_y)))
        # pitch_des = np.sign(B_x - centerpoint) * np.arctan(k * (1 - np.cos(B_x - centerpoint_x)))
        deltas = np.sign(X_B[:2]) * np.arctan(k * (1 - np.cos(X_B[:2] - centerpoint)))
        pitch_des = deltas[0]
        roll_des = deltas[1]
        # yaw_des = np.arctan(centerpoint[1]/centerpoint[0])
        # roll_des = np.arctan(.75*(B_y)**3)
        # pitch_des = -np.arctan(.75*(B_x-0.88)**3)
        yaw_des = 0

        # print(f"Ball: [{B_x}, {B_y}, {X_B[2]}]\nRPY current: [{roll_current}, {pitch_current}, {yaw_current}]\nRPY desired: [{roll_des}, {pitch_des}, {yaw_des}]")
            
        K_p = 4

        dw = K_p*np.array([roll_des-roll_current, pitch_des-pitch_current, yaw_des - yaw_current])
        # print(f"Commanded: {dw}\n")
        # dw = np.zeros_like(dw)
        output.SetFromVector(dw)
