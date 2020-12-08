import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Install pyngrok.
server_args = []
if 'google.colab' in sys.modules:
  server_args = ['--ngrok_http_tunnel']
from meshcat.servers.zmqserver import start_zmq_server_as_subprocess
proc, zmq_url, web_url = start_zmq_server_as_subprocess(server_args=server_args)

from pydrake.all import (
    DiagramBuilder, ConnectMeshcatVisualizer, Simulator, SignalLogger, JacobianWrtVariable, Integrator, 
    LeafSystem, BasicVector, RollPitchYaw, ConstantVectorSource, RigidTransform, SpatialVelocity
)
import time

from utils.station import JugglerStation

class InverseKinematics(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self.plant = plant
        self.plant_context = plant.CreateDefaultContext()
        self.iiwa = plant.GetModelInstanceByName("iiwa7")
        self.P = plant.GetBodyByName("base_link").body_frame()
        self.W = plant.world_frame()

        self.DeclareVectorInputPort("paddle_desired_velocity", BasicVector(6))
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
        output.SetFromVector(np.hstack([0, 0, 0, v_Ball[0], v_Ball[1], -v_Ball[2]))


class Juggler:
    def __init__(self, kp=100, ki=1, kd=20, time_step=0.002, show_axis=False):
        """
        Robotic Kuka IIWA juggler with paddle end effector     

        Args:
            kp (int, optional): proportional gain. Defaults to 100.
            ki (int, optional): integral gain. Defaults to 1.
            kd (int, optional): derivative gain. Defaults to 20.
            time_step (float, optional): time step for internal manipulation station controller. Defaults to 0.002.
            show_axis (boolean, optional): whether or not to show the axis of reflection.
        """        
        self.time = 0
        juggler_station = JugglerStation(
            kp=kp,
            ki=ki,
            kd=kd,
            time_step=time_step,
            show_axis=show_axis)

        self.builder = DiagramBuilder()
        self.station = self.builder.AddSystem(juggler_station.get_diagram())
        self.log = []

        self.visualizer = ConnectMeshcatVisualizer(
            self.builder, output_port=self.station.GetOutputPort("geometry_query"), zmq_url=zmq_url)

        self.plant = juggler_station.get_multibody_plant()

        # ---------------------
        self.ik_sys = self.builder.AddSystem(InverseKinematics(self.plant))
        self.ik_sys.set_name("ik_sys")
        self.mirror = self.builder.AddSystem(VelocityMirror(self.plant))
        self.mirror.set_name("mirror")
        integrator = self.builder.AddSystem(Integrator(7))
        self.builder.Connect(self.station.GetOutputPort("iiwa_position_measured"), self.ik_sys.GetInputPort("iiwa_pos_measured"))
        self.builder.Connect(self.ik_sys.get_output_port(), integrator.get_input_port())
        self.builder.Connect(integrator.get_output_port(), self.station.GetInputPort("iiwa_position"))
        # Useful for debugging
        # desired_vel = self.builder.AddSystem(ConstantVectorSource([0, 0, 0, .1, 0, 0]))
        # self.builder.Connect(desired_vel.get_output_port(), self.ik_sys.GetInputPort("paddle_desired_velocity"))
        self.builder.Connect(self.mirror.get_output_port(), self.ik_sys.GetInputPort("paddle_desired_velocity"))
        self.builder.ExportInput(self.mirror.GetInputPort("ball_pose"), "ball_pose")
        self.builder.ExportInput(self.mirror.GetInputPort("ball_velocity"), "ball_velocity")

        # ---------------------

        self.diagram = self.builder.Build()
        self.simulator = Simulator(self.diagram)
        self.simulator.set_target_realtime_rate(1.0)

        self.context = self.simulator.get_context()
        self.station_context = self.station.GetMyContextFromRoot(self.context)
        self.plant_context = self.plant.GetMyContextFromRoot(self.context)

        # self.plant.SetPositions(self.plant_context, self.plant.GetModelInstanceByName("iiwa7"), [0, np.pi/4, 0, -np.pi/2, 0, -np.pi/4, 0])
        self.plant.SetPositions(self.plant_context, self.plant.GetModelInstanceByName("iiwa7"), [0, np.pi/3, 0, -np.pi/2, 0, -np.pi/3, 0])

        self.station.GetInputPort("iiwa_feedforward_torque").FixValue(self.station_context, np.zeros((7,1)))
        iiwa_model_instance = self.plant.GetModelInstanceByName("iiwa7")
        iiwa_q = self.plant.GetPositions(self.plant_context, iiwa_model_instance)
        integrator.GetMyContextFromRoot(self.context).get_mutable_continuous_state_vector().SetFromVector(iiwa_q)
        
        self.controller = None #TODO: implement LeafSystem-based high level controller


    def command_iiwa_position(self, iiwa_position, simulate=True, duration=0.1, final=True, verbose=False):
        """
        Command the arm to move to a specific configuration

        Args:
            iiwa_position (list): the (length 7 for 7 DoF) list indicating arm joint positions
            simulate (bool, optional): whether or not to visualize the command. Defaults to True.
            duration (float, optional): duration to complete command in simulation. Defaults to 0.1.
            final (bool, optional): whether or not this is the final command in the sequence; relevant for recording. Defaults to True.
            verbose (bool, optional): whether or not to print measured position change. Defaults to False.
        """        
        self.station.GetInputPort("iiwa_position").FixValue(
            self.station_context, iiwa_position)
        
        self.diagram.GetInputPort("paddle_desired_velocity").FixValue(self.context, [0,0,0,0,0,0])
        
        if simulate:
            self.visualizer.start_recording()
            self.simulator.AdvanceTo(self.time + duration)
            self.visualizer.stop_recording()
            
            self.log.append(self.station.GetOutputPort("iiwa_position_measured").Eval(self.station_context))
            
            if verbose:
                print("Commanding position: {}\nMeasured Position: {}\n\n".format(iiwa_position, np.around(self.station.GetOutputPort("iiwa_position_measured").Eval(self.station_context), 3)))
            
            if final:
                self.visualizer.publish_recording()

        self.time += duration


    def t(self, desired, simulate=True, duration=0.1, final=True, verbose=False):
        """
        TODO

        Args:
            simulate (bool, optional): whether or not to visualize the command. Defaults to True.
            duration (float, optional): duration to complete command in simulation. Defaults to 0.1.
            final (bool, optional): whether or not this is the final command in the sequence; relevant for recording. Defaults to True.
            verbose (bool, optional): whether or not to print measured position change. Defaults to False.
        """        
        ball_pose = self.plant.EvalBodyPoseInWorld(self.plant_context, self.plant.GetBodyByName("ball")).translation()
        ball_velocity = self.plant.EvalBodySpatialVelocityInWorld(self.plant_context, self.plant.GetBodyByName("ball"))
        ball_velocity = np.hstack([ball_velocity.rotational(), ball_velocity.translational()])
        # print(ball_velocity)
        self.diagram.GetInputPort("ball_pose").FixValue(self.context, ball_pose)
        self.diagram.GetInputPort("ball_velocity").FixValue(self.context, ball_velocity) 
        if simulate:
            # self.time = round(self.time, 5)
            self.visualizer.start_recording()
            self.simulator.AdvanceTo(self.time + duration)
            self.visualizer.stop_recording()
            
            self.log.append(self.station.GetOutputPort("iiwa_position_measured").Eval(self.station_context))
            
            if verbose:
                print("Time: {}\nMeasured Position: {}\n\n".format(self.time, np.around(self.station.GetOutputPort("iiwa_position_measured").Eval(self.station_context), 3)))
            
            if final:
                self.visualizer.publish_recording()

        self.time += duration




if __name__ == "__main__":
    kp = 300
    ki = 20
    kd = 20
    time_step = .002

    juggler = Juggler(
        kp=kp, 
        ki=ki, 
        kd=kd, 
        time_step=time_step,
        show_axis=True)

    positions = [[r, np.pi/4, 0, -np.pi/2, 0, -np.pi/4, 0 ] for r in np.linspace(0, np.pi, 40)]
    # for i, pos in enumerate(positions):
        # juggler.command_iiwa_position(pos, duration=0.1, final=i==len(positions)-1, verbose=False)
        # juggler.t(desired=[0, 0, 0, 0, 0, 0.1],duration=0.1, final=i==len(positions)-1, verbose=False)
    velocities = [[0, 0, 0, .2*np.cos(t), .2*np.sin(t/1.5), 0] for t in np.linspace(0, 10, 50)]
    velocities = [
        [0, 0, 0, 0, 0, 3], 
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, -1],
        [0, 0, 0, 0, 0, 0]
    ] * 2
    durations = [.15, .1, .15, .1]*2
    # for i, vel in enumerate(velocities):
    #     juggler.t(vel, duration=durations[i], final=i==len(velocities)-1, verbose=True)
    for i in range(20):
        juggler.t(None, duration=0.1, final=i==39, verbose=True)

    df = pd.DataFrame(juggler.log)
    print(df)
    input("\npress enter key to exit...")

    # x = np.linspace(0, np.pi, 40)
    # for i in range(7):
    #     plt.plot(x, df[i], label=i)
    # plt.show()