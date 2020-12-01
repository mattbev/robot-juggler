import sys
import numpy as np

# Install pyngrok.
server_args = []
if 'google.colab' in sys.modules:
  server_args = ['--ngrok_http_tunnel']
from meshcat.servers.zmqserver import start_zmq_server_as_subprocess
proc, zmq_url, web_url = start_zmq_server_as_subprocess(server_args=server_args)

from pydrake.all import (
    Adder, AddMultibodyPlantSceneGraph, AngleAxis, BasicVector, ConnectMeshcatVisualizer, 
    DiagramBuilder, FindResourceOrThrow, Integrator, InverseDynamicsController,
    JacobianWrtVariable, LeafSystem, MultibodyPlant, MultibodyPositionToGeometryPose, 
    Parser, PiecewisePolynomial, PiecewiseQuaternionSlerp, Quaternion, RigidTransform, 
    RollPitchYaw, RotationMatrix, SceneGraph, Simulator, TrajectorySource, PassThrough, Demultiplexer,
    StateInterpolatorWithDiscreteDerivative, SchunkWsgPositionController, MakeMultibodyStateToWsgStateSystem,
    ConnectDrakeVisualizer, AbstractValue, QueryObject
)
from pydrake.examples.manipulation_station import ManipulationStation
from station_builder import make_manipulation_station

def test_manipulation_station_add_iiwa_and_wsg_explicitly():
    station = ManipulationStation()
    parser = Parser(station.get_mutable_multibody_plant(),
                    station.get_mutable_scene_graph())
    plant = station.get_mutable_multibody_plant()

    # Add models for iiwa and wsg
    iiwa_model_file = FindResourceOrThrow(
        "drake/manipulation/models/iiwa_description/iiwa7/"
        "iiwa7_no_collision.sdf")
    iiwa = parser.AddModelFromFile(iiwa_model_file, "iiwa")
    X_WI = RigidTransform.Identity()
    plant.WeldFrames(plant.world_frame(),
                        plant.GetFrameByName("iiwa_link_0", iiwa),
                        X_WI)

    wsg_model_file = FindResourceOrThrow(
        "drake/manipulation/models/wsg_50_description/sdf/"
        "schunk_wsg_50_no_tip.sdf")
    wsg = parser.AddModelFromFile(wsg_model_file, "gripper")
    X_7G = RigidTransform.Identity()
    plant.WeldFrames(
        plant.GetFrameByName("iiwa_link_7", iiwa),
        plant.GetFrameByName("body", wsg))

    # Register models for the controller.
    station.RegisterIiwaControllerModel(
        iiwa_model_file, iiwa, plant.world_frame(),
        plant.GetFrameByName("iiwa_link_0", iiwa), X_WI)
    station.RegisterWsgControllerModel(
        wsg_model_file, wsg,
        plant.GetFrameByName("iiwa_link_7", iiwa),
        plant.GetFrameByName("body", wsg), X_7G)

    # Finalize
    station.Finalize()

    # This WSG gripper model has 2 independent dof, and the IIWA model
    # has 7.
    return station


def test():
    builder = DiagramBuilder()
    station = builder.AddSystem(test_manipulation_station_add_iiwa_and_wsg_explicitly())
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
    visualizer.start_recording()
    simulator.AdvanceTo(5.0)
    visualizer.stop_recording()
    visualizer.publish_recording()


if __name__ == "__main__":
    test()