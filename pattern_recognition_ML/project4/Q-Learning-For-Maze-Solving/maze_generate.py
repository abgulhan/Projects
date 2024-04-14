import argparse
import math

def arg_parse():
    """
    Parses the arguments.
    Returns:
        parser (argparse.ArgumentParser): Parser with arguments.
    """
    parser = argparse.ArgumentParser()
    # General 
    parser.add_argument('--out', type=str, default='results', help='Directory to save maze')
    parser.add_argument('--size', type=str, default='5x3', help='Size of maze in format: h,w . Ex: 3x3')
    parser.add_argument('--start', type=str, default='0,0', help='start coordinates of maze. Format: y,x  Default: 0,0')
    parser.add_argument('--end', type=str, default=None, help='end coordinates of maze. Format: y,x  Default size[0]-1, size[1]-1')
    parser.add_argument('--type', type=str, help='What type of maze to generate. Options: hilbert, rows, cols')
    parser.add_argument('--step', type=int, default=2, help='Size of a step in maze. Use 2 for dqn maze format')


    return parser.parse_args()

class Maze_Draw:
    # size, start and end are tuples with values (height, width)
    def __init__(self, size, start, end, dir='E'):
        self._size = size
        self._start = start
        self._end = end
        self._dir = dir # can have values N,S,E,W
        self._pos = start
        self._maze = [[1]*size[1] for i in range(size[0])]
        self._maze[start[0]][start[1]] = 0
    


    def forward(self, step=1):
        if self._dir == 'N':
            self._pos = [self._pos[0]-1, self._pos[1]]
        elif self._dir == 'S':
            self._pos = [self._pos[0]+1, self._pos[1]]
        elif self._dir == 'E':
            self._pos = [self._pos[0], self._pos[1]+1]
        else: #self._dir == 'W':
            self._pos = [self._pos[0], self._pos[1]-1]
        
        h,w = self._pos
        try:
            self._maze[h][w] = 0
        except:
            print(f"WARNING: Out of maze bounds at coordinate ({h},{w})")

        if step-1 > 0:
            return self.forward(step-1)
        else:
            if self._pos == self._end:
                print("Reached end of maze")
                return False
            return True

    def left(self, parity=1):
        if parity == -1:
            self.right()
        else:     
            if self._dir == 'N':
                self._dir = 'W'
            elif self._dir == 'S':
                self._dir = 'E'
            elif self._dir == 'E':
                self._dir = 'N'
            else: #self._dir == 'W':
                self._dir = 'S'

    def right(self, parity=1):
        if parity == -1:
            self.left()
        else:
            if self._dir == 'N':
                self._dir = 'E'
            elif self._dir == 'S':
                self._dir = 'W'
            elif self._dir == 'E':
                self._dir = 'S'
            else: #self._dir == 'W':
                self._dir = 'N'

    def save_maze(self, file):
        #print(self._maze)
        out = ""
        for row in self._maze:
            out += ','.join(map(str, row))
            out += '\n'
        
        with open(file, 'w') as f:
            f.write(out)
    
def hilbert(maze, level, parity, step):
 
    if level == 0:
        return
 
    maze.right(parity)
    hilbert(maze, level-1, -parity, step)
 
    maze.forward(step)
    maze.left(parity)
    hilbert(maze, level-1, parity, step)
 
    maze.forward(step)
    hilbert(maze, level-1, parity, step)
 
    maze.left(parity)
    maze.forward(step)
    hilbert(maze, level-1, -parity, step)
    maze.right(parity)
 
def rows(maze, height, width):
    maze._dir = 'E'
   
    for i in range(height//2):  
        maze.forward(width-1)
        if i%2 == 0:
            maze.right()
            maze.forward(2)
            maze.right()
        else:
            maze.left()
            maze.forward(2)
            maze.left()
            
def main(args):
    # TODO better argument parsing
    y,x = args.size.split('x')
    size = (int(y), int(x))

    if args.start == None:
        start = (0,0)
    else:
        y,x = args.start.split(',')
        start = (int(y), int(x))

    if args.end == None:
        end = (size[0]-1, size[1]-1)
    else:
        y,x = args.end.split(',')
        end =  (int(y), int(x))
    
    
    if args.type == 'hilbert':
        if (size[0]+1)%2 != 0 or (size[0]+1)%2 != 0:
            print("!!WARNING, hilbert curve size must be size (2^n)-1 for each dimension. May not generate correct maze otherwise")

        level = math.ceil(math.log2(max(size[0], size[1])))-1
        print(f"Hilbert curve level is {level}")
        maze = Maze_Draw(size, start, end, dir='E')
        
        hilbert(maze, level, 1, args.step)
    
    elif args.type == 'rows':
        maze = Maze_Draw(size, start, end, dir='E')
        rows(maze, size[0], size[1])
    
    else:
        print('Error. Unknown type parameter')
        return
    
    
    maze.save_maze(f'{args.out}/maze-{args.type}-{size[0]}x{size[1]}.csv')
 
 
if __name__ == '__main__':
    args = arg_parse()
    main(args)