import os
import sys
import time

import requests

from tools import (
    FINISHED,
    get_action_descv2,
    get_action_from_views_actions,
    parse_views,
)

####because 'AGENTENV_PATH' is set so environment can be found
sys.path.insert(0, os.environ.get("AGENTENV_PATH"))
from environment import AndroidController

# config
AVD_NAME = "Copy2_of_p6a"
TASK_METADATA_PATH = "../dataset/llamatouch_task_metadata.tsv"
# TASK_METADATA_URL = "https://raw.githubusercontent.com/LlamaTouch/LlamaTouch/main/dataset/llamatouch_task_metadata.tsv"
emulator_controller_args = {
        "snapshot" : "default_boot",
        "port" : "5558",        # If port is occupied, please switch to 5556, 5558... and so forth
        "no-window" : "true",  # Change this to "true" to run the emulator without GUI.
    }
first_n_episodes=int(os.environ.get("FIRST_N_EPISODES", 495))
range_pair = (38, 495)
# response = requests.get(TASK_METADATA_URL)
# with open(TASK_METADATA_PATH, "w") as f:
#     f.write(response.content)


def run_on_agentenv():  

    ac = AndroidController(
        avd_name=AVD_NAME,
        emulator_controller_args=emulator_controller_args,
        local_output_path="exec_output_llamatouch_autodroid_deepseek",
        max_steps=30,
        instruction_fp=TASK_METADATA_PATH,
    )

    # setup AgentEnv
    ac.set_up()
    
    for index in range(first_n_episodes):
        try:
            task_description,_,_,_,full_path = ac.get_instruction()# get instruction from AgentEnv
            if task_description is None:
                continue       
            print(f"Current instruction: {task_description}")
            if(index+1 < range_pair[0]): # 放在ac.get_instruction() iterator后面
                continue
            elif index+1 > range_pair[1]:
                break
            try_count = 0
            while try_count < 5: # try at most 5 times for each task
                try:
                    try_count += 1  
                    # setup task environment if needed
                    print(f"setting up task {task_description}...")
                    ac.setup_task(task_description) # some tasks need to setup preparation before execution
                    
                    # go to the dropdown s
                    print(f"swipe up the screen")
                    ac.device.swipe(500, 1500, 500, 500) #upstairs, x then y
                    
                    time.sleep(2)

                    action_history = [f"- start from the home screen"]
                    thought_history = [f"my goal is to finish the task {task_description}"]
                    ac_conversation = ""
                    while not ac.episode_done():
                        ac_conversation = ""
                        # get view_hierarchy_json from AgentEnv
                        raw_views = ac.get_state()["view_hierarchy_json"]
                        views = parse_views(raw_views)

                        s = time.time()
                        # get autodroid agent action
                        action, candidate_actions, target_view, thought, prompt, response = (
                            get_action_from_views_actions(
                                task_description=task_description,
                                views=views,
                                action_history=action_history,
                                thought_history=thought_history,
                            )
                        )
                        ac_conversation += f"-------------------------prompt asking for text input in  SetTextEvent:-------------------------\n{prompt}\n" + f"-------------------------Response:-------------------------\n{response}\n"
                        ac.save_chat(ac_conversation)
                        
                        print("-----------------------------------------------------------------------------")
                        print(f"propmt: {prompt}")
                        print(f"got the action from the agent, costed time: {time.time()-s};action: {action}")
                        print(f"candidate_actions: {candidate_actions}")
                        print(f"target_view: {target_view}")
                        print(f"thought: {thought}")

                        if action == FINISHED:
                            ac.post_action(
                                "action_type: STATUS_TASK_COMPLETE, touch_point: [-1.0, -1.0], lift_point: [-1.0, -1.0], typed_text: ''"
                            )
                            break

                        action_history.append(get_action_descv2(action, target_view))
                        thought_history.append(thought)

                        if action.event_type == "key":
                            if action.name == "BACK":
                                ac.post_action(
                                    "action_type: PRESS_BACK, touch_point: [-1.0, -1.0], lift_point: [-1.0, -1.0], typed_text: ''" ,do_execute=True
                                )

                        elif action.event_type == "click":
                            tl, br = action.view["bounds"]
                            ac.tap(tl, br, do_execute=True)

                        elif action.event_type == "long_click":
                            tl, br = action.view["bounds"]
                            ac.long_press(tl, br, do_execute=True)

                        elif action.event_type == "swipe":
                            act = f"action_type: dual_point, touch_point: [{action.start_x}, {action.start_y}], lift_point: [{action.end_x}, {action.end_y}], typed_text: ''"
                            ac.post_action(act, do_execute=True)

                        elif action.event_type == "set_text":
                            ac.text(action.text, do_execute=True)

                        else:
                            raise Exception(f"Error action event type: {action.event_type}")
                    
                    # save the last environment state of an episode
                    ac.get_state()
                    # reset the environment for next task
                    ac.reset_env()
                    
                    break

                except Exception as e:
                    print(f"Error in task {task_description}: {e}")
                    # remove content in folder os.path.join("exec_output", "captured_data")
                    os.system(f"mv -r {full_path} {os.path.join(ac.local_output_path, 'error_episode', f'{index}_{try_count}')}")
                    import traceback
                    traceback.print_exc()

                    # reset the environment and go to next task
                    ac.reset_env()
                    continue
    
        except Exception as e:
            print(f"Error and skip task {task_description}: {e}")
            import traceback
            traceback.print_exc()
            # reset the environment and go to next task
            ac.reset_env()
            continue
        
    # Execution finished, close AgentEnv
    ac.tear_down()


if __name__ == "__main__":
    run_on_agentenv()
