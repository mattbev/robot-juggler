import sys
import numpy as np

# Install pyngrok.
server_args = []
if 'google.colab' in sys.modules:
  server_args = ['--ngrok_http_tunnel']
from meshcat.servers.zmqserver import start_zmq_server_as_subprocess
proc, zmq_url, web_url = start_zmq_server_as_subprocess(server_args=server_args)


from pydrake.systems.meshcat_visualizer import MeshcatVisualizer
from pydrake.all import (
    Adder, AddMultibodyPlantSceneGraph, ConnectMeshcatVisualizer, DiagramBuilder, 
    InverseDynamicsController, MultibodyPlant, Parser, SceneGraph, Simulator, 
    PassThrough, Demultiplexer, StateInterpolatorWithDiscreteDerivative, 
    SchunkWsgPositionController, MakeMultibodyStateToWsgStateSystem, Integrator,
    RigidTransform, RollPitchYaw
)
from manipulation.scenarios import AddIiwa, AddWsg, AddRgbdSensors
from manipulation.utils import FindResource

class JugglerStation:
    def __init__(self, kp=100, ki=1, kd=20, time_step=0.002, show_axis=False):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.time_step = time_step
        self.show_axis = show_axis
        self.diagram, self.plant = self.make_manipulation_station(self.kp, self.ki, self.kd, self.time_step, self.show_axis)

    def get_multibody_plant(self):
        return self.plant

    def get_diagram(self):
        return self.diagram

    @staticmethod
    def make_manipulation_station(kp=100, ki=1, kd=20, time_step=0.002, show_axis=False):
        """
        Create the juggler manipulation station.

        Args:
            kp (int, optional): proportional gain. Defaults to 100.
            ki (int, optional): integral gain. Defaults to 1.
            kd (int, optional): derivative gain. Defaults to 20.
            time_step (float, optional): controller time step. Defaults to 0.002.

        Returns:
            (tuple[(diagram), (plant)]): the diagram and plant
        """        
        builder = DiagramBuilder()

        # Add (only) the iiwa, WSG, and cameras to the scene.
        plant, scene_graph = AddMultibodyPlantSceneGraph(
            builder, time_step=time_step)
        iiwa = AddIiwa(plant, collision_model="with_box_collision")
        # wsg = AddWsg(plant, iiwa)
        parser = Parser(plant)
        parser.AddModelFromFile(
            FindResource("models/camera_box.sdf"), "camera0")
        parser.AddModelFromFile("utils/models/floor.sdf")
        parser.AddModelFromFile("utils/models/paddle.sdf")
        parser.AddModelFromFile("utils/models/ball.sdf")
        plant.WeldFrames(plant.GetFrameByName("iiwa_link_7"), plant.GetFrameByName("base_link"), RigidTransform(RollPitchYaw(0, np.pi/2, 0), [0, 0, 0.25]))
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
        controller_plant = MultibodyPlant(time_step=time_step)
        controller_iiwa = AddIiwa(controller_plant, collision_model="with_box_collision")
        # AddWsg(controller_plant, controller_iiwa, welded=True)
        controller_plant.Finalize()

        # Add the iiwa controller
        iiwa_controller = builder.AddSystem(
            InverseDynamicsController(
                controller_plant,
                kp=[kp]*num_iiwa_positions,
                ki=[ki]*num_iiwa_positions,
                kd=[kd]*num_iiwa_positions,
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

        # Wsg controller.
        # wsg_controller = builder.AddSystem(SchunkWsgPositionController())
        # wsg_controller.set_name("wsg_controller")
        # builder.Connect(wsg_controller.get_generalized_force_output_port(),             
        #                 plant.get_actuation_input_port(wsg))
        # builder.Connect(plant.get_state_output_port(wsg), wsg_controller.get_state_input_port())
        # builder.ExportInput(wsg_controller.get_desired_position_input_port(), "wsg_position")
        # builder.ExportInput(wsg_controller.get_force_limit_input_port(), "wsg_force_limit")
        # wsg_mbp_state_to_wsg_state = builder.AddSystem(
        #     MakeMultibodyStateToWsgStateSystem())
        # builder.Connect(plant.get_state_output_port(wsg), wsg_mbp_state_to_wsg_state.get_input_port())
        # builder.ExportOutput(wsg_mbp_state_to_wsg_state.get_output_port(), "wsg_state_measured")
        # builder.ExportOutput(wsg_controller.get_grip_force_output_port(), "wsg_force_measured")

        # Cameras.
        AddRgbdSensors(builder, plant, scene_graph)

        # Export "cheat" ports.
        builder.ExportOutput(scene_graph.get_query_output_port(), "geometry_query")
        builder.ExportOutput(plant.get_contact_results_output_port(), "contact_results")
        builder.ExportOutput(plant.get_state_output_port(), "plant_continuous_state")

        diagram = builder.Build()

        return diagram, plant


def station_test():
    builder = DiagramBuilder()
    station = builder.AddSystem(JugglerStation().get_diagram())

    visualizer = ConnectMeshcatVisualizer(
        builder, output_port=station.GetOutputPort("geometry_query"), zmq_url=zmq_url)

    diagram = builder.Build()
    simulator = Simulator(diagram)

    context = simulator.get_context()
    station_context = station.GetMyContextFromRoot(context)

    station.GetInputPort("iiwa_position").FixValue(station_context, [0, np.pi/2, 0, -np.pi/2, 0, -np.pi/4, 0])
    
    visualizer.start_recording()
    simulator.AdvanceTo(5.0)
    visualizer.stop_recording()
    visualizer.publish_recording()
    


if __name__ == "__main__":
    station_test()