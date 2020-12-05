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
    LeafSystem, BasicVector, RollPitchYaw
)

from utils.station import JugglerStation

class InverseKinematics(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self.plant = plant
        self.plant_context = plant.CreateDefaultContext()
        self.iiwa = plant.GetModelInstanceByName("iiwa7")
        self.P = plant.GetBodyByName("base_link").body_frame()
        self.W = plant.world_frame()

        self.DeclareVectorInputPort("paddle_desired_pose", BasicVector(6))
        self.DeclareVectorInputPort("iiwa_pos_measured", BasicVector(7))
        self.DeclareVectorOutputPort("iiwa_velocity", BasicVector(7), self.CalcOutput)
        self.iiwa_start = plant.GetJointByName("iiwa_joint_1").velocity_start()
        self.iiwa_end = plant.GetJointByName("iiwa_joint_7").velocity_start()

    def CalcOutput(self, context, output):
        q = self.GetInputPort("iiwa_pos_measured").Eval(context)
        X_P = self.GetInputPort("paddle_desired_pose").Eval(context)
        self.plant.SetPositions(self.plant_context, self.iiwa, q)
        J_P = self.plant.CalcJacobianSpatialVelocity(
            self.plant_context, JacobianWrtVariable.kQDot, 
            self.P, [0,0,0], self.plant.world_frame(), self.plant.world_frame())
        J_P = J_P[:,self.iiwa_start:self.iiwa_end+1]
        
        X_P_0 = self.plant.EvalBodyPoseInWorld(self.plant_context, self.plant.GetBodyByName("base_link"))

        rpy = RollPitchYaw(X_P_0.rotation()).vector()
        xyz = X_P_0.translation()
        diff = X_P - np.hstack([rpy, xyz])

        k = 0.2
        V_P_desired = k * diff / np.linalg.norm(diff)
        v = np.around(np.linalg.pinv(J_P).dot(V_P_desired), 8)
        # #overwrite for debugging
        v = [np.pi/4,0,0,0,0,0,0]
        output.SetFromVector(v)



class Juggler:
    def __init__(self, kp=100, ki=1, kd=20, time_step=0.002):
        """
        Robotic Kuka IIWA juggler with paddle end effector     

        Args:
            kp (int, optional): proportional gain. Defaults to 100.
            ki (int, optional): integral gain. Defaults to 1.
            kd (int, optional): derivative gain. Defaults to 20.
            time_step (float, optional): time step for internal manipulation station controller. Defaults to 0.002.
        """        
        self.time = 0
        juggler_station = JugglerStation(
            kp=kp,
            ki=ki,
            kd=kd,
            time_step=time_step)

        self.builder = DiagramBuilder()
        self.station = self.builder.AddSystem(juggler_station.get_diagram())
        self.log = []

        self.visualizer = ConnectMeshcatVisualizer(
            self.builder, output_port=self.station.GetOutputPort("geometry_query"), zmq_url=zmq_url)

        self.plant = juggler_station.get_multibody_plant()

        # ---------------------
        self.ik_sys = self.builder.AddSystem(InverseKinematics(self.plant))
        self.ik_sys.set_name("ik_sys")
        integrator = self.builder.AddSystem(Integrator(7))
        self.builder.Connect(self.station.GetOutputPort("iiwa_position_measured"), self.ik_sys.GetInputPort("iiwa_pos_measured"))
        self.builder.Connect(self.ik_sys.get_output_port(), integrator.get_input_port())
        self.builder.Connect(integrator.get_output_port(), self.station.GetInputPort("iiwa_position"))
        self.builder.ExportInput(self.ik_sys.GetInputPort("paddle_desired_pose"), "paddle_desired_pose")
        # ---------------------

        self.diagram = self.builder.Build()
        self.simulator = Simulator(self.diagram)
        self.simulator.set_target_realtime_rate(1.0)

        self.context = self.simulator.get_context()
        self.station_context = self.station.GetMyContextFromRoot(self.context)
        self.plant_context = self.plant.GetMyContextFromRoot(self.context)

        self.controller = None #TODO: implement LeafSystem-based high level controller

        # self.plant.SetPositions(self.plant_context, self.iiwa, q)
        # self.station.GetInputPort("iiwa_position").FixValue(self.station_context, [0, np.pi/4, 0, -np.pi/2, 0, -np.pi/4, 0 ])


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
        
        self.diagram.GetInputPort("paddle_desired_pose").FixValue(self.context, [0,0,0,0,0,0])
        
        if simulate:
            self.visualizer.start_recording()
            self.simulator.AdvanceTo(self.time + duration)
            self.visualizer.stop_recording()
            
            self.log.append(self.station.GetOutputPort("iiwa_position_measured").Eval(self.station_context))
            
            if verbose:
                print("Commanding position: {}\nMeasured Position: {}\n\n".format(iiwa_position, self.station.GetOutputPort("iiwa_position_measured").Eval(self.station_context)))
            
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

        self.diagram.GetInputPort("paddle_desired_pose").FixValue(self.context, desired)
        
        if simulate:
            self.time = round(self.time, 5)
            # print(f"time: {self.time}")
            # print(self.context)
            self.visualizer.start_recording()
            self.simulator.AdvanceTo(self.time + duration)
            self.visualizer.stop_recording()
            
            self.log.append(self.station.GetOutputPort("iiwa_position_measured").Eval(self.station_context))
            
            if verbose:
                print("Time: {}\nMeasured Position: {}\n\n".format(self.time, self.station.GetOutputPort("iiwa_position_measured").Eval(self.station_context)))
            
            if final:
                self.visualizer.publish_recording()

        self.time += duration




if __name__ == "__main__":
    kp = 100
    ki = 1
    kd = 20
    time_step = .002

    juggler = Juggler(
        kp=kp, 
        ki=ki, 
        kd=kd, 
        time_step=time_step)

    positions = [[r, np.pi/4, 0, -np.pi/2, 0, -np.pi/4, 0 ] for r in np.linspace(0, np.pi, 40)]
    for i, pos in enumerate(positions):
        # juggler.command_iiwa_position(pos, duration=0.1, final=i==len(positions)-1, verbose=False)
        juggler.t(desired=[0, 0, 0, -(i/40)**2+1.5, i/40, 0],duration=0.1, final=i==len(positions)-1, verbose=False)

    df = pd.DataFrame(juggler.log)
    print(df)
    input("\npress enter key to exit...")

    # x = np.linspace(0, np.pi, 40)
    # for i in range(7):
    #     plt.plot(x, df[str(i)], label=i)
    # plt.show()