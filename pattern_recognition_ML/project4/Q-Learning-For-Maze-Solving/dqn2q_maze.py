import numpy as np
import argparse

def arg_parse():
    """
    Parses the arguments.
    Returns:
        parser (argparse.ArgumentParser): Parser with arguments.
    """
    parser = argparse.ArgumentParser()
    # General 
    parser.add_argument('--maze', type=str, help='Name of input maze')
    parser.add_argument('--out', type=str, default='results', help='Directory to save maze')

    return parser.parse_args()

# transform maze from dqn fromat into classic q-learning maze format
def dqn2q(input_maze_file, output_file):
    input_maze = np.genfromtxt(input_maze_file, delimiter=',').astype(int)
    shape = input_maze.shape
    #print(input_maze)
    out = np.zeros(shape, dtype=int)
    #(1=North, 2=East, 4=South, 8=West)
    N,E,S,W = 1,2,4,8
    for i in range(shape[0]):
        for j in range(shape[1]):
            if input_maze[i][j]==1:
                continue
            if i != 0 and input_maze[i-1][j]==0:
                out[i][j] += N
            if i != shape[0]-1 and input_maze[i+1][j]==0:
                out[i][j] += S
            if j != 0 and input_maze[i][j-1]==0:
                out[i][j] += W
            if j != shape[1]-1 and input_maze[i][j+1]==0:
                out[i][j] += E
    #print(out)
    
    # Note that the code expects it a certain way so we pre-process it here.
    out = np.swapaxes(out, 0, 1)
    np.save(output_file, out)
            
    
if __name__ == '__main__':
    args = arg_parse()
    dqn2q(args.maze, args.out)