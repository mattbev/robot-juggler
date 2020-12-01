import pydot
import sys
import importlib
import numpy as np
from urllib.request import urlretrieve
from IPython.display import display, SVG

# Install pyngrok.
server_args = []
if 'google.colab' in sys.modules:
  server_args = ['--ngrok_http_tunnel']
from meshcat.servers.zmqserver import start_zmq_server_as_subprocess
proc, zmq_url, web_url = start_zmq_server_as_subprocess(server_args=server_args)


from pydrake.examples.manipulation_station import ManipulationStation
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer
from pydrake.all import (
    Adder, AddMultibodyPlantSceneGraph, AngleAxis, BasicVector, ConnectMeshcatVisualizer, 
    DiagramBuilder, FindResourceOrThrow, Integrator, InverseDynamicsController,
    JacobianWrtVariable, LeafSystem, MultibodyPlant, MultibodyPositionToGeometryPose, 
    Parser, PiecewisePolynomial, PiecewiseQuaternionSlerp, Quaternion, RigidTransform, 
    RollPitchYaw, RotationMatrix, SceneGraph, Simulator, TrajectorySource, PassThrough, Demultiplexer,
    StateInterpolatorWithDiscreteDerivative, SchunkWsgPositionController, MakeMultibodyStateToWsgStateSystem,
    ConnectDrakeVisualizer, AbstractValue, QueryObject
)
from trajectories import (
    make_paddle_position_trajectory, make_paddle_orientation_trajectory
)

from manipulation.scenarios import AddIiwa, AddWsg, AddRgbdSensors
from manipulation.utils import FindResource

def make_manipulation_station(time_step=0.002):
    builder = DiagramBuilder()

    # Add (only) the iiwa, WSG, and cameras to the scene.
    plant, scene_graph = AddMultibodyPlantSceneGraph(
        builder, time_step=time_step)
    iiwa = AddIiwa(plant)
    # wsg = AddWsg(plant, iiwa)
    parser = Parser(plant, scene_graph)
    parser.AddModelFromFile(
        FindResource("models/camera_box.sdf"), "camera0")
    parser.AddModelFromFile("paddle.sdf")
        
    plant.WeldFrames(plant.GetFrameByName("iiwa_link_7"), plant.GetFrameByName("base_link"))
    plant.Finalize()

    num_iiwa_positions = plant.num_positions(iiwa)

    # I need a PassThrough system so that I can export the input port.
    iiwa_position = builder.AddSystem(PassThrough(num_iiwa_positions))
    builder.ExportInput(iiwa_position.get_input_port(), "iiwa_position")
    builder.ExportOutput(iiwa_position.get_output_port(), "iiwa_position_command")

    # Export the iiwa "state" outputs.
    demux = builder.AddSystem(Demultiplexer(
        2 * num_iiwa_positions, num_iiwa_positions))
    builder.Connect(plant.get_state_output_port(iiwa), demux.get_input_port())
    builder.ExportOutput(demux.get_output_port(0), "iiwa_position_measured")
    builder.ExportOutput(demux.get_output_port(1), "iiwa_velocity_estimated")
    builder.ExportOutput(plant.get_state_output_port(iiwa), "iiwa_state_estimated")

    # Make the plant for the iiwa controller to use.
    # TODO: I should *NOT* need to add this system to the diagram, but I was having "keep-alive" issues.  Must fix.
    controller_plant = builder.AddSystem(MultibodyPlant(time_step=time_step))
    controller_iiwa = AddIiwa(controller_plant)
    controller_plant.Finalize()

    # Add the iiwa controller
    iiwa_controller = builder.AddSystem(
        InverseDynamicsController(
            controller_plant,
            kp=[100]*num_iiwa_positions,
            ki=[1]*num_iiwa_positions,
            kd=[20]*num_iiwa_positions,
            has_reference_acceleration=False))
    iiwa_controller.set_name("iiwa_controller")
    builder.Connect(
        plant.get_state_output_port(iiwa), iiwa_controller.get_input_port_estimated_state())

    # Add in the feed-forward torque
    adder = builder.AddSystem(Adder(2, num_iiwa_positions))
    builder.Connect(iiwa_controller.get_output_port_control(),
                    adder.get_input_port(0))
    # Use a PassThrough to make the port optional (it will provide zero values if not connected).
    torque_passthrough = builder.AddSystem(
        PassThrough([0]*num_iiwa_positions))
    builder.Connect(torque_passthrough.get_output_port(), adder.get_input_port(1))
    builder.ExportInput(torque_passthrough.get_input_port(), "iiwa_feedforward_torque")
    builder.Connect(adder.get_output_port(), plant.get_actuation_input_port(iiwa))

    # Add discrete derivative to command velocities.
    desired_state_from_position = builder.AddSystem(
        StateInterpolatorWithDiscreteDerivative(
            num_iiwa_positions, time_step, suppress_initial_transient=True))
    desired_state_from_position.set_name("desired_state_from_position")
    builder.Connect(desired_state_from_position.get_output_port(),      
                    iiwa_controller.get_input_port_desired_state())
    builder.Connect(iiwa_position.get_output_port(), desired_state_from_position.get_input_port())

    # Export commanded torques.
    #builder.ExportOutput(adder.get_output_port(), "iiwa_torque_commanded")
    #builder.ExportOutput(adder.get_output_port(), "iiwa_torque_measured")

    # Cameras.
    AddRgbdSensors(builder, plant, scene_graph)

    # Export "cheat" ports.
    builder.ExportOutput(scene_graph.get_query_output_port(), "geometry_query")
    builder.ExportOutput(plant.get_contact_results_output_port(), "contact_results")
    builder.ExportOutput(plant.get_state_output_port(), "plant_continuous_state")

    diagram = builder.Build()
    return diagram


def test():
    builder = DiagramBuilder()
    station = builder.AddSystem(make_manipulation_station())

    visualizer = ConnectMeshcatVisualizer(
        builder, output_port=station.GetOutputPort("geometry_query"), zmq_url=zmq_url)

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    visualizer.load(visualizer.GetMyContextFromRoot(context))
    diagram.Publish(context)

    simulator = Simulator(diagram)
    station_context = station.GetMyContextFromRoot(simulator.get_mutable_context())
    station.GetInputPort("iiwa_feedforward_torque").FixValue(station_context, np.zeros((7,1)))
    # station.GetInputPort("iiwa_position").FixValue(station_context, [0, np.pi/4, 0, -np.pi/2, 0, -np.pi/4, 0])
    # integrator.GetMyContextFromRoot(simulator.get_mutable_context()).get_mutable_continuous_state_vector().SetFromVector(station.GetIiwaPosition(station_context))
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(.01)

    simulator.AdvanceTo(5.0)

    while True: continue

# test()


