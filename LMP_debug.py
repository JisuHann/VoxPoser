import os
import time
from robocasa_controller import RoboCasaController
from src.interfaces import setup_LMP
from src.LMP import merge_dicts, exec_safe
# with open('exec_safe_inputs.txt', 'r') as f:
#     exec_safe_inputs = f.read()

# {'np': <module 'numpy' from '/usr/local/lib/python3.10/dist-packages/numpy/__init__.py'>, 'euler2quat': <function euler2quat at 0x7faf92bf1cf0>, 'quat2euler': <function quat2euler at 0x7faf92bf1d80>, 'qinverse': <function qinverse at 0x7faf92bf1360>, 'qmult': <function qmult at 0x7faf92bf1120>, 'cm2index': <bound method LMP_interface.cm2index of <interfaces.LMP_interface object at 0x7faa7cd48190>>, 'detect': <bound method LMP_interface.detect of <interfaces.LMP_interface object at 0x7faa7cd48190>>, 'execute': <bound method LMP_interface.execute of <interfaces.LMP_interface object at 0x7faa7cd48190>>, 'get_ee_pos': <bound method LMP_interface.get_ee_pos of <interfaces.LMP_interface object at 0x7faa7cd48190>>, 'get_empty_affordance_map': <bound method LMP_interface.get_empty_affordance_map of <interfaces.LMP_interface object at 0x7faa7cd48190>>, 'get_empty_avoidance_map': <bound method LMP_interface.get_empty_avoidance_map of <interfaces.LMP_interface object at 0x7faa7cd48190>>, 'get_empty_gripper_map': <bound method LMP_interface.get_empty_gripper_map of <interfaces.LMP_interface object at 0x7faa7cd48190>>, 'get_empty_rotation_map': <bound method LMP_interface.get_empty_rotation_map of <interfaces.LMP_interface object at 0x7faa7cd48190>>, 'get_empty_velocity_map': <bound method LMP_interface.get_empty_velocity_map of <interfaces.LMP_interface object at 0x7faa7cd48190>>, 'index2cm': <bound method LMP_interface.index2cm of <interfaces.LMP_interface object at 0x7faa7cd48190>>, 'pointat2quat': <bound method LMP_interface.pointat2quat of <interfaces.LMP_interface object at 0x7faa7cd48190>>, 'reset_to_default_pose': <bound method LMP_interface.reset_to_default_pose of <interfaces.LMP_interface object at 0x7faa7cd48190>>, 'set_voxel_by_radius': <bound method LMP_interface.set_voxel_by_radius of <interfaces.LMP_interface object at 0x7faa7cd48190>>, 'parse_query_obj': <LMP.LMP object at 0x7faa38688df0>, 'get_affordance_map': <LMP.LMP object at 0x7faa38688e80>, 'get_avoidance_map': <LMP.LMP object at 0x7faa38688f10>, 'get_velocity_map': <LMP.LMP object at 0x7faa38688fa0>, 'get_rotation_map': <LMP.LMP object at 0x7faa38689030>, 'get_gripper_map': <LMP.LMP object at 0x7faa386890c0>, 'composer': <LMP.LMP object at 0x7faa38688c70>, 'exec': <function exec_safe.<locals>.<lambda> at 0x7faa7cd6aef0>, 'eval': <function exec_safe.<locals>.<lambda> at 0x7faa7cd6aef0>}

exec_safe_inputs_dir = "exec_safe_LMP_gvars.pkl"
LMP_EXEC_INSTANCES=[{
    "task": "ArrangeVegetables",
    "exec_safe_inputs":[
    """
def ret_val():
    # objects = ['knife', 'vegetable2', 'vegetable1', 'dishwasher', 'sink', 'coffee']
    
    gripper = detect('gripper')
    return gripper""",
    """
def ret_val():
    
    affordance_map = get_empty_affordance_map()
    vegetable1 = parse_query_obj('vegetable1')
    x, y, z = vegetable1.position
    affordance_map[x, y, z] = 1
    return affordance_map""",
    """
def ret_val():
    # objects = ['sink', 'vegetable1', 'vegetable2', 'dishwasher', 'knife']
    
    vegetable1 = detect('vegetable1')
    return vegetable1""",
    """
def ret_val():
    
    gripper_map = get_empty_gripper_map()
    # open everywhere
    gripper_map[:, :, :] = 1
    # close when 1cm around the vegetable1
    vegetable1 = parse_query_obj('vegetable1')
    set_voxel_by_radius(gripper_map, vegetable1.position, radius_cm=1, value=0)
    return gripper_map""",
    """
def ret_val():
    # objects = ['sink', 'vegetable1', 'vegetable2', 'dishwasher', 'knife']
    
    vegetable1 = detect('vegetable1')
    return vegetable1""",
    ]
},
]

def main(): 
    # with open(exec_safe_inputs_dir, 'rb') as f:
    #     gvars = pickle.load(f)
    for lmp_name in LMP_EXEC_INSTANCES:
        controller = RoboCasaController(env_name=lmp_name['task'], execute_task_with_lmp=False)
        controller.load_task()
        controller.lmps, controller.lmp_env = setup_LMP(controller.env, controller.config, debug=controller.debug)
        gvars = merge_dicts([controller.lmps['plan_ui']._fixed_vars, controller.lmps['plan_ui']._variable_vars])
        lvars = {}
        empty_fn = lambda *args, **kwargs: None
        custom_gvars = merge_dicts([
            gvars,
            {'exec': empty_fn, 'eval': empty_fn}
        ])
        for exec_safe_input in lmp_name['exec_safe_inputs']:
            file_name = f"<generated:{int(time.time())}>"
            print('==================Execute==================')
            print(exec_safe_input)
            compiled = compile(exec_safe_input, file_name, "exec")
            exec(compiled, custom_gvars, lvars)
            ret_val = lvars['ret_val']()
            print("----------Output--------------")
            print(ret_val)
if __name__ == '__main__':
    main()


