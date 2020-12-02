import sys
import numpy as np

# Install pyngrok.
server_args = []
if 'google.colab' in sys.modules:
  server_args = ['--ngrok_http_tunnel']
from meshcat.servers.zmqserver import start_zmq_server_as_subprocess
proc, zmq_url, web_url = start_zmq_server_as_subprocess(server_args=server_args)

from pydrake.all import (
    DiagramBuilder, ConnectMeshcatVisualizer, Simulator
)

from utils.station import make_manipulation_station

class Juggler:
    def __init__(self, time_step=0.002):
        """
        Robotic Kuka IIWA juggler with paddle end effector

        Args:
            time_step (float, optional): time step for internal manipulation station controller. Defaults to 0.002.
        """        
        self.time = 0
        self.builder = DiagramBuilder()
        self.station = self.builder.AddSystem(make_manipulation_station(time_step))
        self.visualizer = ConnectMeshcatVisualizer(
            self.builder, output_port=self.station.GetOutputPort("geometry_query"), zmq_url=zmq_url)
        self.diagram = self.builder.Build()
        self.simulator = Simulator(self.diagram)
        self.context = self.simulator.get_context()
        self.station_context = self.station.GetMyContextFromRoot(self.context)

        self.controller = None #TODO: implement LeafSystem-based high level controller
        
    def command_position(self, iiwa_position, visualize=True, duration=1.0, final=True):
        """
        Command the arm to move to a specific configuration

        Args:
            iiwa_position (list): the (length 7 for 7 DoF) list indicating arm joint positions
            visualize (bool, optional): whether or not to visualize the command. Defaults to True.
            duration (float, optional): duration to complete command in simulation. Defaults to 1.0.
            final (bool, optional): whether or not this is the final command in the sequence; relevant for recording. Defaults to True.
        """        
        self.station.GetInputPort("iiwa_position").FixValue(
            self.station_context, iiwa_position)
        
        if visualize:
            self.visualizer.start_recording()
            self.simulator.AdvanceTo(self.time + duration)
            self.visualizer.stop_recording()
            
            if final:
                self.visualizer.publish_recording()

        self.time += duration




if __name__ == "__main__":
    juggler = Juggler()
    juggler.command_position([0, np.pi/2, 0, -np.pi/2, 0, -np.pi/4, 0], final=False)
    juggler.command_position([0, 0, 0,0, 0, 0, 0])
