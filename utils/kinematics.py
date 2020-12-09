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
        # v = np.linalg.pinv(J_P).dot(np.hstack([w_P_desired, v_P_desired]))
        v = np.linalg.pinv(J_P).dot(np.hstack([[0, 0, 0], v_P_desired]))

        # #overwrite for debugging
        # v = [0,0,0,0,0,0,0]
        output.SetFromVector(v)


class VelocityMirror(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self.plant = plant
        self.plant_context = plant.CreateDefaultContext()
        self.iiwa = plant.GetModelInstanceByName("iiwa7")
        # self.Ball = plant.GetBodyByName("ball")
        self.Paddle = plant.GetBodyByName("base_link")
        self.W = plant.world_frame()

        self.DeclareVectorInputPort("iiwa_pos_measured", BasicVector(7))
        self.DeclareVectorInputPort("iiwa_velocity_estimated", BasicVector(7))
        self.DeclareVectorInputPort("ball_pose", BasicVector(3))
        self.DeclareVectorInputPort("ball_velocity", BasicVector(3))
        self.DeclareVectorOutputPort("mirror_velocity", BasicVector(3), self.CalcOutput)

    def CalcOutput(self, context, output):
        q = self.GetInputPort("iiwa_pos_measured").Eval(context)
        q_dot = self.GetInputPort("iiwa_velocity_estimated").Eval(context)
        p_Ball = np.array(self.GetInputPort("ball_pose").Eval(context))
        p_Ball[2] = 0
        # p_Ball[2] = 2 * 0.75 - p_Ball[2]
        v_Ball = np.array(self.GetInputPort("ball_velocity").Eval(context))
        if v_Ball[2] >= 0:
            v_Ball[2] = -1*v_Ball[2]
        else:
            v_Ball[2] = -1*v_Ball[2]   
        self.plant.SetPositionsAndVelocities(self.plant_context, self.iiwa, np.hstack([q, q_dot]))
        p_Paddle = np.array(self.plant.EvalBodyPoseInWorld(self.plant_context, self.Paddle).translation())
        p_Paddle[2] = 0
        v_Paddle = np.array(self.plant.EvalBodySpatialVelocityInWorld(self.plant_context, self.Paddle).translational())
        
        K_p = 3
        K_d = 1
        v_P_desired = K_p*(p_Ball - p_Paddle) + K_d*(v_Ball - v_Paddle)
        v_P_desired[2] = .8*v_Ball[2]

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
        # print(roll_current, pitch_current, R_P[2])
        B_x, B_y = X_B[0], X_B[1]
        roll_nom, pitch_nom, yaw_nom = -np.pi, 0, np.pi                
                
        roll_des = roll_nom + np.arctan(6*(B_y)**5)
        pitch_des = pitch_nom - np.arctan(6*(B_x-0.88)**5)
        yaw_des = yaw_nom

        # print(f"Ball: [{B_x}, {B_y}]\nRPY current: [{roll_current}, {pitch_current}, {yaw_current}]\nRPY desired: [{roll_des}, {pitch_des}, {yaw_des}]\n")
            
        k = 1
        dw = k*np.array([roll_des-roll_current, pitch_des-pitch_current, yaw_des - yaw_current])
        # print(f"Assigned: {dw}\n\n")
        # dw = np.zeros_like(dw)
        output.SetFromVector(dw)
