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
from trajectories import (
    make_paddle_position_trajectory, make_paddle_orientation_trajectory
)

class SimpleContinuousTimeSystem(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        
        self.DeclareContinuousState(1)             # One state variable.
        self.DeclareVectorOutputPort("y", BasicVector(1), self.CopyStateOut)           # One output.

    # xdot(t) = -x(t) + x^3(t)
    def DoCalcTimeDerivatives(self, context, derivatives):
        x = context.get_continuous_state_vector().GetAtIndex(0)
        xdot = -x + x**3
        derivatives.get_mutable_vector().SetAtIndex(0, xdot)

    # y = x
    def CopyStateOut(self, context, output):
        x = context.get_continuous_state_vector().CopyToVector()
        output.SetFromVector(x)


class SimpleDiscreteTimeSystem(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        
        self.DeclareDiscreteState(1)             # One state variable.
        self.DeclareVectorOutputPort("y", BasicVector(1), self.CopyStateOut)           # One output.
        self.DeclarePeriodicDiscreteUpdate(1.0)  # One second timestep.

    # x[n+1] = x^3[n]
    def DoCalcDiscreteVariableUpdates(self, context, events, discrete_state):
        x = context.get_discrete_state_vector().GetAtIndex(0)
        xnext = x**3
        discrete_state.get_mutable_vector().SetAtIndex(0, xnext)

    # y = x
    def CopyStateOut(self, context, output):
        x = context.get_discrete_state_vector().CopyToVector()
        output.SetFromVector(x)


class PseudoInverseController(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa = plant.GetModelInstanceByName("iiwa")
        self._Paddle = plant.GetFrameByName("base_link")
        self._W = plant.world_frame()

        self.w_Paddle_port = self.DeclareVectorInputPort("omega_WPaddle", BasicVector(3))
        self.v_Paddle_port = self.DeclareVectorInputPort("v_WPaddle", BasicVector(3))
        self.q_port = self.DeclareVectorInputPort("iiwa_position", BasicVector(7))
        self.DeclareVectorOutputPort("iiwa_velocity", BasicVector(7), 
                                     self.CalcOutput)
        # TODO(russt): Add missing binding
        #joint_indices = plant.GetJointIndices(self._iiwa)
        #self.position_start = plant.get_joint(joint_indices[0]).position_start()
        #self.position_end = plant.get_joint(joint_indices[-1]).position_start()
        self.iiwa_start = plant.GetJointByName("iiwa_joint_1").velocity_start()
        self.iiwa_end = plant.GetJointByName("iiwa_joint_7").velocity_start()

    def CalcOutput(self, context, output):
        w_Paddle = self.w_Paddle_port.Eval(context)
        v_Paddle = self.v_Paddle_port.Eval(context)
        V_Paddle = np.hstack([w_Paddle, v_Paddle])
        q = self.q_port.Eval(context)
        self._plant.SetPositions(self._plant_context, self._iiwa, q)
        J_Paddle = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context, JacobianWrtVariable.kV, 
            self._Paddle, [0,0,0], self._W, self._W)
        J_Paddle = J_Paddle[:,self.iiwa_start:self.iiwa_end+1] # Only iiwa terms.
        v = np.linalg.pinv(J_Paddle).dot(V_Paddle)
        output.SetFromVector(v)


builder = DiagramBuilder()

station = builder.AddSystem(ManipulationStation())
station.SetupClutterClearingStation()
# station.Finalize()

# plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-4)

plant = station.get_mutable_multibody_plant()
scene_graph = station.get_mutable_scene_graph()

# Note that we parse into both the plant and the scene_graph here.
parser = Parser(plant, scene_graph)
# iiwa_model = parser.AddModelFromFile(
#         FindResourceOrThrow("drake/manipulation/models/iiwa_description/sdf/iiwa14_no_collision.sdf"))
# paddle_model = parser.AddModelFromFile("paddle.sdf")
# plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))

paddle_model = parser.AddModelFromFile("paddle.sdf")
plant.WeldFrames(plant.GetFrameByName("iiwa_link_7"), plant.GetFrameByName("base_link"))
# plant.Finalize()
station.Finalize()

# temp_context = station.CreateDefaultContext()
# temp_plant_context = plant.GetMyContextFromRoot(temp_context)

X_Paddle = {"initial": RigidTransform(RotationMatrix.MakeXRotation(-np.pi/2.0), [0, 0, 1]), 
            "intermediate": RigidTransform(RotationMatrix.MakeXRotation(-np.pi/2.0), [-0.5, 0, 1]),
            "final": RigidTransform(RotationMatrix.MakeXRotation(-np.pi/2.0), [-1, 0, 1])}

times = {"initial": 0, "intermediate": 3, "final": 6}

# Make the trajectories
traj_p_Paddle = make_paddle_position_trajectory(X_Paddle, times)
traj_v_Paddle = traj_p_Paddle.MakeDerivative()
traj_R_Paddle = make_paddle_orientation_trajectory(X_Paddle, times)
traj_w_Paddle = traj_R_Paddle.MakeDerivative()

v_Paddle_source = builder.AddSystem(TrajectorySource(traj_v_Paddle))
v_Paddle_source.set_name("v_WPaddle")
w_Paddle_source = builder.AddSystem(TrajectorySource(traj_w_Paddle))
w_Paddle_source.set_name("omega_WPaddle")
controller = builder.AddSystem(PseudoInverseController(plant))
controller.set_name("PseudoInverseController")
builder.Connect(v_Paddle_source.get_output_port(), controller.GetInputPort("v_WPaddle"))
builder.Connect(w_Paddle_source.get_output_port(), controller.GetInputPort("omega_WPaddle"))

print(f"\n{station.GetInputPort('iiwa_position')}\n")


integrator = builder.AddSystem(Integrator(7))
integrator.set_name("integrator")
builder.Connect(controller.get_output_port(), 
                integrator.get_input_port())
builder.Connect(integrator.get_output_port(),
                station.GetInputPort("iiwa_position"))
builder.Connect(station.GetOutputPort("iiwa_position_measured"),
                controller.GetInputPort("iiwa_position"))

# traj_wsg_command = make_wsg_command_trajectory(times)
# wsg_source = builder.AddSystem(TrajectorySource(traj_wsg_command))
# wsg_source.set_name("wsg_command")
# builder.Connect(wsg_source.get_output_port(), station.GetInputPort("wsg_position"))

meshcat = ConnectMeshcatVisualizer(builder,
    station.get_scene_graph(),
    output_port=station.GetOutputPort("pose_bundle"),
#    delete_prefix_on_load=False,  # Use this if downloading is a pain.
    zmq_url=zmq_url,
)

diagram = builder.Build()
diagram.set_name("random_movement")
temp_context = diagram.CreateDefaultContext()
temp_plant_context = plant.GetMyContextFromRoot(temp_context)

simulator = Simulator(diagram)
station_context = station.GetMyContextFromRoot(simulator.get_mutable_context())
# TODO(russt): Add this missing python binding
#integrator.set_integral_value(
#    integrator.GetMyContextFromRoot(simulator.get_mutable_context()), 
#        station.GetIiwaPosition(station_context))
integrator.GetMyContextFromRoot(simulator.get_mutable_context()).get_mutable_continuous_state_vector().SetFromVector(station.GetIiwaPosition(station_context))

simulator.set_target_realtime_rate(1.0)
meshcat.start_recording()
simulator.AdvanceTo(traj_p_Paddle.end_time())
meshcat.stop_recording()
meshcat.publish_recording()