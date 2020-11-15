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

X_Paddle = {"initial": RigidTransform(RotationMatrix.MakeXRotation(-np.pi/2.0), [0, -0.25, 0.25]), 
            "intermediate": RigidTransform(RotationMatrix.MakeXRotation(-np.pi/2.0), [0, -1, 0.25]),
            "final": RigidTransform(RotationMatrix.MakeXRotation(-np.pi/2.0), [0, -1.75, 0.25])}

times = {"initial": 0, "intermediate": 3, "final": 6}

def make_paddle_position_trajectory(X_Paddle, times):
    """
    Constructs a gripper position trajectory from the plan "sketch".

    Args:
        X_Paddle ([type]): [description]
        times ([type]): [description]

    Returns:
        [type]: [description]
    """ 
    traj = PiecewisePolynomial.FirstOrderHold(
        [times["initial"], times["intermediate"]], np.vstack([X_Paddle["initial"].translation(), X_Paddle["intermediate"].translation()]).T)

    # TODO(russt): I could make this less brittle if I was more careful on the names above, and just look up the pose for every time (in order)
    traj.AppendFirstOrderSegment(times["final"], X_Paddle["final"].translation())

    return traj



def make_paddle_orientation_trajectory(X_Paddle, times):
    """
    Constructs a gripper position trajectory from the plan "sketch".

    Args:
        X_Paddle ([type]): [description]
        times ([type]): [description]

    Returns:
        [type]: [description]
    """ 
    traj = PiecewiseQuaternionSlerp()
    traj.Append(times["initial"], X_Paddle["initial"].rotation())
    traj.Append(times["intermediate"], X_Paddle["intermediate"].rotation())
    traj.Append(times["final"], X_Paddle["final"].rotation())

    return traj

