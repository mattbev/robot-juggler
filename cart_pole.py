import math
import numpy as np
import matplotlib.pyplot as plt

from pydrake.examples.manipulation_station import ManipulationStation
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer
from pydrake.all import (
    Adder, AddMultibodyPlantSceneGraph, AngleAxis, BasicVector, ConnectMeshcatVisualizer, 
    DiagramBuilder, FindResourceOrThrow, Integrator, InverseDynamicsController, DirectCollocation,
    JacobianWrtVariable, LeafSystem, MultibodyPlant, MultibodyPositionToGeometryPose, 
    Parser, PiecewisePolynomial, PiecewiseQuaternionSlerp, Quaternion, RigidTransform, 
    RollPitchYaw, RotationMatrix, SceneGraph, Simulator, TrajectorySource, PassThrough, Demultiplexer,
    StateInterpolatorWithDiscreteDerivative, SchunkWsgPositionController, MakeMultibodyStateToWsgStateSystem,
    ConnectDrakeVisualizer, AbstractValue, QueryObject, Solve
)
from manipulation.scenarios import AddIiwa
from manipulation.utils import FindResource


builder = DiagramBuilder()
# station = builder.AddSystem(make_manipulation_station(.01))
# plant = station.get_multibody_plant() #this doesn't work
# =======
plant, scene_graph = AddMultibodyPlantSceneGraph(
        builder, time_step=0.0)
iiwa = AddIiwa(plant)
# wsg = AddWsg(plant, iiwa)
parser = Parser(plant, scene_graph)
parser.AddModelFromFile(
    FindResource("models/camera_box.sdf"), "camera0")
parser.AddModelFromFile("paddle.sdf")
plant.WeldFrames(plant.GetFrameByName("iiwa_link_7"), plant.GetFrameByName("base_link"))
plant.mutable_gravity_field().set_gravity_vector([0, 0, 0])
plant.Finalize()

builder.Connect(scene_graph.get_query_output_port(), plant.get_geometry_query_input_port())
builder.Build()
plant_context = plant.CreateDefaultContext()
plant.GetPositions(plant_context)

dircol = DirectCollocation(
    system=plant,
    context=plant_context,
    input_port_index=plant.get_actuation_input_port(iiwa).get_index(),
    assume_non_continuous_states_are_fixed=True,
    num_time_samples=100,
    minimum_timestep=0.01,
    maximum_timestep=0.1,
)

dircol.AddEqualTimeIntervalsConstraints()
initial_state = (1, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, -1.2, 0, 1.6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
# initial_state = (0, np.pi/4, 0, -np.pi/2, 0, -np.pi/4, 0)
dircol.AddBoundingBoxConstraint(initial_state, initial_state,
                                dircol.initial_state())
# More elegant version is blocked by drake #8315:
# dircol.AddLinearConstraint(dircol.initial_state() == initial_state)

final_state = (1, 1, 1, 0, 0, 0, 0, 1, 1, 0, -1.2, 0, 1.6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
dircol.AddBoundingBoxConstraint(final_state, final_state, dircol.final_state())
# dircol.AddLinearConstraint(dircol.final_state() == final_state)

R = 10  # Cost on input "effort".
u = dircol.input()
dircol.AddRunningCost(R * u[0]**2)

# Add a final cost equal to the total duration.
dircol.AddFinalCost(dircol.time())

# initial_x_trajectory = PiecewisePolynomial.FirstOrderHold(
#     [0., 4.], np.column_stack((initial_state, final_state)))  # yapf: disable
# dircol.SetInitialTrajectory(PiecewisePolynomial(), initial_x_trajectory)

print("SOLVING.....")
result = Solve(dircol)
assert result.is_success()
print("SOLVED!!")

fig, ax = plt.subplots()

u_trajectory = dircol.ReconstructInputTrajectory(result)
times = np.linspace(u_trajectory.start_time(), u_trajectory.end_time(), 100)
u_lookup = np.vectorize(u_trajectory.value)
u_values = u_lookup(times)

ax.plot(times, u_values)
ax.set_xlabel("time (seconds)")
ax.set_ylabel("force (Newtons)")

ax.show()