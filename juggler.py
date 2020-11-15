import sys
import numpy as np

# Install pyngrok.
server_args = []
if 'google.colab' in sys.modules:
  server_args = ['--ngrok_http_tunnel']
from meshcat.servers.zmqserver import start_zmq_server_as_subprocess
proc, zmq_url, web_url = start_zmq_server_as_subprocess(server_args=server_args)

from pydrake.examples.manipulation_station import ManipulationStation
from pydrake.all import (
    AddMultibodyPlantSceneGraph, AngleAxis, BasicVector, ConnectMeshcatVisualizer, 
    DiagramBuilder, FindResourceOrThrow, Integrator, InverseDynamicsController,
    JacobianWrtVariable, LeafSystem, MultibodyPlant, MultibodyPositionToGeometryPose, 
    Parser, PiecewisePolynomial, PiecewiseQuaternionSlerp, Quaternion, RigidTransform, 
    RollPitchYaw, RotationMatrix, SceneGraph, Simulator, TrajectorySource
)
from dynamics import (
    SimpleContinuousTimeSystem, SimpleDiscreteTimeSystem, PseudoInverseController
)


class Juggler():
    def __init__(self):
        builder = DiagramBuilder()

        # Find a better way to do this...
        station = ManipulationStation()
        station.SetupClutterClearingStation()
        station.Finalize()

        builder.AddSystem(station)

        # Adds both MultibodyPlant and the SceneGraph, and wires them together.
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-4)

        # plant = station.get_mutable_multibody_plant()
        # scene_graph = station.get_mutable_scene_graph()
        
        # Note that we parse into both the plant and the scene_graph here.
        parser = Parser(plant, scene_graph)
        iiwa_model = parser.AddModelFromFile(
                FindResourceOrThrow("drake/manipulation/models/iiwa_description/sdf/iiwa14_no_collision.sdf"))
        paddle_model = parser.AddModelFromFile("paddle.sdf")
        plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))
        plant.WeldFrames(plant.GetFrameByName("iiwa_link_7"), plant.GetFrameByName("base_link"))

        plant.Finalize()

        # Adds the MeshcatVisualizer and wires it to the SceneGraph.
        self.meshcat = ConnectMeshcatVisualizer(builder, scene_graph, zmq_url=zmq_url, delete_prefix_on_load=True)

        # PID Control
        Kp = np.full(7, 100)
        Ki = 2 * np.sqrt(Kp)
        Kd = np.full(7, 1)
        iiwa_controller = builder.AddSystem(InverseDynamicsController(plant, Kp, Ki, Kd, False))
        iiwa_controller.set_name("iiwa_controller")
        builder.Connect(plant.get_state_output_port(iiwa_model),
                        iiwa_controller.get_input_port_estimated_state())
        builder.Connect(iiwa_controller.get_output_port_control(),
                        plant.get_actuation_input_port())

        # create system
        # self.system = builder.AddSystem(SimpleDiscreteTimeSystem())

        self.diagram = builder.Build()

        self.context = self.diagram.CreateDefaultContext()
        plant_context = plant.GetMyMutableContextFromRoot(self.context)

        q0 = np.array([0, np.pi/4, 0, -np.pi/2, 0, -np.pi/4, 0])
        x0 = np.hstack((q0, [0, 0, 0, 0, 0, 0, 0]))
        plant.SetPositions(plant_context, q0)
        iiwa_controller.GetInputPort('desired_state').FixValue(
            iiwa_controller.GetMyMutableContextFromRoot(self.context), x0)



    def simulate(self, duration=5.0):
        """
        Simulates the juggler

        Args:
            duration (float, optional): Number of timesteps. Defaults to 5.0.
        """        
        simulator = Simulator(self.diagram, self.context)
        simulator.set_target_realtime_rate(1.0)

        self.meshcat.start_recording()
        simulator.AdvanceTo(duration)
        self.meshcat.stop_recording()
        self.meshcat.publish_recording()


if __name__ == "__main__":
    juggler = Juggler()
    juggler.simulate()
