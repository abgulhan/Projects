Changes to code:

Modified maze_view_2d.py and maze_env.py to be able to set goal location. When adding environment to maze_env.py, use the optional parameter goal to set a different ending location in maze (ex: goal=np.array((6,0))) 

Modified dqn/deep_q_learning.py: Added type of maze '--maze_type' as an additional command line parameter. This changes the end location if set as 'hilbert' (--maze_type hilbert)

Added maze_generate.py - generates a maze in dqn format
Added dqn2q_maze.py - converts a dqn maze format into q_learning maze format


=======================================
How to Generate New Mazes:

Note: Generated mazes are stored in the maze_generate folder. Mazes with .npy extension are in classical q-learning maze format and .csv are in deep q-learning format.

To generate Hilbert curve maze for dqn (size should be 2^n-1 for each dimension):
python .\maze_generate.py --out ./my_mazes --type hilbert --size 7x7
python .\maze_generate.py --out ./my_mazes --type hilbert --size 15x15
...

Note: Hilbert curve mazes have an ending point of (0, width-1). This should be specified when running dqn. Use the parameter '--type hilbert' when running deep_q_learning.py

To generate row maze for dqn:
python .\maze_generate.py --out ./my_mazes --type rows --size 7x7
python .\maze_generate.py --out ./my_mazes --type rows --size 15x15

To convert a maze in dqn format into q_learning format use:
python .\dqn2q_maze.py --maze <input maze in dqn format> --out <output location of maze in q_learing format>
Ex: python .\dqn2q_maze.py --maze .\my_mazes\maze-rows-15x15.csv --out .\my_mazes\maze-rows-15x15.npy


=======================================
Running DQN Solvers:

To run Hilbert mazes in dqn, we need to change the ending location. To do thus use parameter --maze_type hilbert
Ex: python deep_q_learning.py --maze_csv maze_samples/maze-hilbert-15x15.csv --config_file config.cfg --maze_type hilbert

The names of the new mazes are:
maze_samples/maze-hilbert-7x7.csv
maze_samples/maze-hilbert-15x15.csv
maze_samples/maze-rows-7x7.csv
maze_samples/maze-rows-10x10.csv
maze_samples/maze-rows-15x15.csv

Note: The solved_threshold parameter should be given appropriately in config.cfg for each maze type, otherwise training does not stop when a correct solution is found. Some values for maze files are listed below.

maze-sample-10x10.cfg : solved_threshold = 19
maze-hilbert-7x7.cfg : solved_threshold = 31
maze-hilbert-15x15.cfg : solved_threshold = 128
maze-rows-7x7.cfg : solved_threshold = 25
maze-rows-10x10.cfg : solved_threshold = 55
maze-rows-15x15.cfg : solved_threshold = 113


Running Q_learning Solvers:

The ending location for Hilbert curve mazes is determined with the 'goal' parameter in maze_env.py  No changes need to be done when running deep_q_learning.py, other than changing the config file ENV_NAME for the new maze environment.

The registered ENV_NAME's are:
maze-hilbert-7x7
maze-hilbert-15x15
maze-rows-7x7
maze-rows-10x10
maze-rows-15x15