import sys
import numpy as np

# For set of constraints of the form Ax<=b, input parameter
# \p Ab is A|b where A matrix and b vector are concatenated.
# Returns a tuple with two np.matrix elements, in which the
# first element is an A'|b' matrix, where A' and b' are the 
# result of projecting the last free variable in x (which 
# corresponds to the second last column in \p Ab) and adding 
# the new bounds. The second element is a matrix that contains
# only the upper and lower bounds of y, with y coefficient 
# of 1 or -1 of the form A|b.
def FME_project(Ab: np.matrix):# -> tuple[np.matrix, np.matrix]:
    height, width = Ab.shape
    #print("input value:")
    #print(Ab)
    if (height < 1 or width < 2):
        print(Ab)
        print("Invalid input matrix dimensions")
        return
    
    Ab = Ab.astype(float) #make sure all elements are floats
    
    upper_bounds = []#numpy.matrix
    lower_bounds = []#numpy.matrix
    zero_bounds = []#numpy.matrix
    for row in range(height): #make coefficients of last variable 1, -1 or 0
        if (Ab[row, width-2] == 0):
            #zero_bounds = np.append(zero_bounds, Ab[row], axis=0)
            zero_bounds.append(np.squeeze(np.asarray(Ab[row])))
            pass;
        elif (Ab[row, width-2] > 0):
            Ab[row] = Ab[row]/Ab[row, width-2] #last free variable located at 2nd last column
            #lower_bounds = np.append(lower_bounds, Ab[row], axis=0)
            upper_bounds.append(np.squeeze(np.asarray(Ab[row])))
        else: #Ab[row, width-2] < 0
            Ab[row] = Ab[row]/(-Ab[row, width-2])
            #upper_bounds = np.append(upper_bounds, Ab[row], axis=0)
            lower_bounds.append(np.squeeze(np.asarray(Ab[row])))
            
    #print("After reducing coefficients")
    #print(Ab)
    if (len(upper_bounds) == 0):
        new_upper = np.zeros(width)
        new_upper[width-2] = 1;
        new_upper[width-1] = np.inf
        upper_bounds.append(new_upper)
    if (len(lower_bounds) == 0):
        new_lower = np.zeros(width)
        new_lower[width-2] = -1;
        new_lower[width-1] = np.inf
        lower_bounds.append(new_lower)


        
    #print(upper_bounds)
    #print(lower_bounds)
    #generating new bounds
    new_bounds = []
    for l in upper_bounds:
        for u in lower_bounds:
            new_bounds.append(np.squeeze(np.asarray(l+u)))
    
    #print("new bounds:")
    #print(np.matrix(new_bounds))
    #
    #building matrix with new constrains and without projected variable
    #and also removing duplicate columns (note: this changes the row order)
    Ab_new = np.matrix(np.unique((new_bounds+zero_bounds), axis=0))
    #print("Ab temp")
    #print(Ab_new)
    Ab_new = np.delete(Ab_new, width-2, axis=1) #delete column of projected variable
    
    #building matrix for lower and upper bounds of projected variable
    var_bounds = np.matrix(np.unique((lower_bounds+ upper_bounds), axis=0))
        
    #print("new constraints after projecting")
    #print(Ab_new)
    #print("/n")
    #print("bounds for projected variable")
    #print(var_bounds)
    
    #removing duplicate rows
    new_array = np.unique([tuple(row) for row in Ab_new])
    uniques = (new_array)

    #print(set(lower_bounds+upper_bounds))
    #print("============================")
    return (Ab_new, var_bounds)
    

# Calculates if bounds are valid for last free variable 'v' 
# in \p bounds. Input parameter \p bounds is of the form 
# A|b (same format as input for FME_project()) and \p values is a list of any
# valid values for the free variables (x in Ax<b) other
# than the last free variable. If there is no free variables
# (when width of bounds == 2), then \p values is an empty list ([])
# Returns the maximum valid value for v given \p values for 
# the other free variables, as a float. Also return if bounds
# are valid or not as a tuple (bool, float).
# For example if x in Ax <= b contains [a,b,c,d], then \p
# this function will calculate bounds for d using values in
# \p values = (a_value, b_value, c_value)
def check_bounds(bounds: np.matrix, values: list): #-> tuple[bool, float]:
    #print("called check_bounds()")
    #print("bounds")
    #print(bounds)
    #print()
    #print("values")
    #print(values)
    num_vars = bounds.shape[1] - 1
    num_rows = bounds.shape[0]
    
    height, width = bounds.shape
    
    upper_bounds = []
    lower_bounds = []
    
    
    
    for row in range(num_rows):
        if (bounds[row, width-2] == 0): #if bound variable has 0 coefficient
            pass;
        elif (bounds[row, width-2] > 0): 
            upper_bounds.append(np.squeeze(np.asarray(bounds[row])))
        else: #Ab[row, width-2] < 0
            lower_bounds.append(np.squeeze(np.asarray(bounds[row])))
    
    
    #print(upper_bounds)
    #print(lower_bounds)
    upper = []
    lower = []
    if (num_vars > 1):
        #calculate upper bounds
        A = np.delete(np.matrix(upper_bounds), [-1,-2], axis=1) 
        vals = np.c_[values] #transform values into column vector
        b = np.matrix(upper_bounds)[:,-1] #get last column of upper bounds
        #print(A)
        #print(vals)
        #print(np.matmul(A, vals))
        #print(b)
        upper = b - np.matmul(A, vals)
        
        #calculate lower bounds
        A = np.delete(np.matrix(lower_bounds), [-1,-2], axis=1) #remove last two colums from lower_bounds
        vals = np.c_[values] #transform values into column vector
        b = np.matrix(lower_bounds)[:,-1] #get last column of lower bounds
        lower = np.matmul(A, vals) - b

    else: # if only one variable
        upper = np.matrix(upper_bounds)[:,-1]
        lower = (np.matrix(lower_bounds)[:,-1]) * -1
        
    #print("=========")
    #print(upper)
    #print(lower)
    min_upper = min(upper).item(0)
    max_lower = max(lower).item(0)
    #print("===max lower===")
    #print(max_lower)
    #print("===min upper===")
    #print(min_upper)
    #print(max_lower == min_upper)
    valid = True
    
    equal = (abs(min_upper - max_lower) <= 1e-09) #check if floating points are almost equal, since floating points aren't exact
    if (not equal and max_lower > min_upper): #floating points may have small amount of error
        valid = False
    #print(valid)
    retval = 0
    if (min_upper == np.inf and max_lower == -np.inf):
        retval = 0
    elif (min_upper == np.inf):
        retval = max_lower
    else:
        retval = min_upper
        
    return (valid, retval)

def FME(Ab: np.matrix): # -> bool:
    bounds_list = []
    while (Ab.shape[1] > 1):
        Ab_new, bounds_new = FME_project(Ab)
        Ab = Ab_new
        bounds_list.append(bounds_new)
        
    #print(bounds_list)
    
    values = []
    x = 1
    for bounds in reversed(bounds_list):
        #print(f"BOUNDS FOR X{x}")
        x+=1
        valid, new_val = check_bounds(bounds, values)
        if (not valid): #check if bounds are valid
            return False
        values.append(new_val)
        
    return True    
    
    
    
    
    
if __name__ == '__main__':
    filename = "sample1.txt"
    out_fname = "sample1_output_solution.txt"
    
    write_output = True
    n = len(sys.argv)-1

    if (n==0):
        print("usage is ./python3 FME.py <input file> <output file[optional]>")
        quit()
    if (n==1):
        filename = str(sys.argv[1])
        write_output = False
    elif (n>=2):
        filename = str(sys.argv[1])
        out_fname = str(sys.argv[2])

    
    data = np.loadtxt(filename, dtype=int)
    Ab = np.matrix(data)

    #Ab = np.matrix('1,0,6;1,1,9;1,-1,5;-2,1,-7') #sample 1
    #Ab = np.matrix('1,1,1,10;1,-1,2,20;2,-1,-1,-1;-1,1,-1,5') #sample 2
    #Ab = np.matrix('1,-4,2;1,5,7;-1,0,-3') #sample 3
    #Ab = np.matrix('1,2,3,9;2,3,4,20;8,-1,0,20') #sample 4
    #Ab = np.matrix('1,1,0,0,0,7;0,1,1,0,0,9;0,0,1,1,0,11;0,0,0,1,1,13') #sample 5
    #Ab = np.matrix('1,0,10;-1,0,-10.1;0,1,1;0,-2,1') #false sample
    #Ab = np.matrix('1,2,3,4;2,1,3,10;3,2,1,20') # sample
    #while (Ab.shape[1] > 1): #while there are more than 1 column in matrix a
    result = FME(Ab)
    output = ""
    if result:
        output = f"solution exists for {filename}"
    else:
        output = f"solution does not exist for {filename}"
    print(output)
    
    
        #Write result to file
    if (write_output):
        out_file = open(out_fname, "w+")
        
        out_file.write(output)
        out_file.close()

