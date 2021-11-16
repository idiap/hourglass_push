- Change "user_definitions.py" according to your needs.

- This code has been tested with V-REP_PRO_EDU_V3_6_0_Ubuntu16_04.

- The variable "train_preset_folder" indicates the directory where training scenarios are stored. Some training scenarios can be found here:

https://drive.google.com/drive/folders/1lht7gRVPZxw3Hhxga36qlfMTvKP_qxKg?usp=sharing

- The mask and the push_into_box networks, which use the HourGlass architecture are defined in "PushingSuccessPredictor.py" and "Push_Into_Box_Net.py", respectively.

- To use this code:

1) Open V-REP
2) Inside your V-REP folder, in "remoteApiConnections.txt", set "portIndex1_port = 19992" or change "vrep_port" in "user_definitions.py"
3) In VREP, click "File", "Open scene" and choose the scene "HOURGLASS_PUSH_CODE/HourGlass_push_online_RL/simulation/simulation.ttt"
4) Run "python main.py --is_sim --is_training --training_preset_cases --load_snapshot"

- For any questions, please contact

marco.ewerton@idiap.ch



