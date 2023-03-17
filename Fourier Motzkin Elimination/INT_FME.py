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
# Let value to be projected = p
# While making new bound with an upper and lower bound, the 
# coefficients of p are multiplied with an integer such that
# they become the least common multiple
def INT_FME_project(Ab: np.matrix): #-> tuple[np.matrix, np.matrix]:
    height, width = Ab.shape
    #print("input value:")
    #print(Ab)
    if (height < 1 or width < 2):
        print(Ab)
        print("Invalid input matrix dimensions")
        return
    
    #Ab = Ab.astype(int) #make sure all elements are integers
    
    upper_bounds = []
    lower_bounds = []
    zero_bounds = []
    
    #calculating LCM for projection variables
    #lcm_arr = np.squeeze(np.asarray(Ab[:,width-2])) #get column of projection variable coefficients
    #lcm_arr = lcm_arr[lcm_arr != 0] #remove 0 elements
    #lcm = abs(np.lcm.reduce(lcm_arr)) #calculate LCM
    
    #print(f"LCM = {lcm}")
    
    for row in range(height): #make coefficients of last variable equal to LCM
        if (Ab[row, width-2] == 0):
            zero_bounds.append(np.squeeze(np.asarray(Ab[row])))
            continue
            
        if (Ab[row, width-2] > 0):
            Ab[row] = Ab[row] #last free variable located at 2nd last column
            upper_bounds.append(np.squeeze(np.asarray(Ab[row])))
        else: #Ab[row, width-2] < 0
            Ab[row] = Ab[row]
            lower_bounds.append(np.squeeze(np.asarray(Ab[row])))
            
    #print("After multiplying coefficients")
    #print(Ab)
    if (len(upper_bounds) == 0):
        new_upper = np.zeros(width)
        new_upper[width-2] = int(1);
        new_upper[width-1] = np.inf
        upper_bounds.append(new_upper)
    if (len(lower_bounds) == 0):
        new_lower = np.zeros(width)
        new_lower[width-2] = int(-1);
        new_lower[width-1] = np.inf
        lower_bounds.append(new_lower)


        
    #print(upper_bounds)
    #print(lower_bounds)
    #generating new bounds
    new_bounds = []
    for l in upper_bounds:
        for u in lower_bounds:
            #finding LCM and multiplying each bound

            
            coeff_l = l[width-2]
            coeff_u = u[width-2]
            #print(f"coeff_l {coeff_l}   coeff_u {coeff_u}")
            if (abs(coeff_l) == np.inf or abs(coeff_u) == np.inf):
                mult_l = np.inf
                mult_u = np.inf
                #print(f"LCM inf")
            else:
                lcm = abs(np.lcm(int(coeff_l), int(coeff_u)))
                #print(f"LCM {lcm}")
                mult_l = abs(lcm//coeff_l)
                mult_u = abs(lcm//coeff_u)
                
            new_bounds.append(np.squeeze(np.asarray(l*mult_l + u*mult_u)))
    
    #print("new bounds:")
    #print(np.matrix(new_bounds))
    
    #building matrix with new constrains and without projected variable
    #and also removing duplicate columns (note: this changes the row order)
    Ab_new = np.matrix(np.unique((new_bounds+zero_bounds), axis=0))
    Ab_new = np.delete(Ab_new, width-2, axis=1) #delete column of projected variable
    
    #building matrix for lower and upper bounds of projected variable
    var_bounds = np.matrix(np.unique((lower_bounds+ upper_bounds), axis=0))
        
    #print("new constraints after projecting")
    #print(Ab_new)
    #print()
    #print("bounds for projected variable")
    #print(var_bounds)
    #print("===============================")
    
    #removing duplicate rows
    new_array = np.unique([tuple(row) for row in Ab_new])
    uniques = (new_array)

    #print(set(lower_bounds+upper_bounds))
    
    return (Ab_new, var_bounds)
    
def generate_loop_nest(bounds_list, output_file = ""):
    result = "" #stores result
    indentation = "    " #what to use for indentation 
    depth = 0 #keeps track of for loop depth
    
    variable = "x_" #what variable to print in for loop

    #print(bounds_list)
    var_names = [variable+str(i+1) for i in range(len(bounds_list))]
    for bounds in bounds_list:
        is_exact = True #keeps track of exactness or inexactness of projection
        height, width = bounds.shape
        upper_bounds = []
        lower_bounds = []

        #print(bounds)
        
        lcm_arr = np.squeeze(np.asarray(bounds[:,-2])) #get column of projection variable coefficients
        if ((np.inf in lcm_arr) or (-np.inf in lcm_arr)):
            lcm = np.inf
            #print("lcm is inf")
            is_exact = False
        else:   
            lcm_arr = lcm_arr.astype(int) #make sure every element is of int type. Required for lcm
            #print(lcm_arr)
            
            #lcm_arr = lcm_arr[lcm_arr != 0] #remove 0 elements 
            lcm = int(abs(np.lcm.reduce(lcm_arr))) #calculate LCM
            #print(f"LCM {lcm}")
            if (lcm != 1):
                is_exact = False
                
        
        
        for row in range(height): #get upper and lower bounds
            if (bounds[row, width-2] == 0): #if bound variable has 0 coefficient
                pass;
                
            mult = abs(lcm/bounds[row, width-2]) #use to multiply coefficients to that boundary variable has same coefficient magnitude
            if (bounds[row, width-2] > 0): 
                upper_bounds.append(np.squeeze(np.asarray(bounds[row]) * mult))
            else: #Ab[row, width-2] < 0
                lower_bounds.append(np.squeeze(np.asarray(bounds[row]) * mult))

        #print(lower_bounds)
        #print(upper_bounds)
        result += indentation*depth
        result += "FOR "
        if (len(lower_bounds) == 0 or len(upper_bounds) == 0):
            print("ERROR")
            return ("ERROR")
        #check if we need MAX(...) for lower bounds or MIN(...) for upper bounds   
        no_max = False
        no_min = False
        
        if (len(lower_bounds) == 1):
            no_max = True
        if (len(upper_bounds) == 1):
            no_min = True
        
        if (not no_max):
            result += "MAX("

        for row in lower_bounds:
            b = row[-1] # last element is constant b
            if (b == np.inf):
                result += str(-b)
            else:
                result += str(int(-b))
            for var in range(len(row)-1):
                expr = ""
                coeff = row[var]
                if coeff == 0:
                    continue
                if (var == depth):
                    continue
                result += " + " + str(int(row[var])) + "*" + var_names[var]
            if (not no_max and var != len(row)-1):
                result += ", "
        if (not no_max and var != len(row)-1):
            result = result[:-2] #remove comma at end of string           
        if (not no_max):
            result += ")"
            
        result += " <= " + str(abs(int(lower_bounds[0][depth]))) + "*" + var_names[depth] + " <= " 


        if (not no_min):
            result += "MIN("
        
        for row in upper_bounds:
            b = row[-1] # last element is constant b
            if (b == np.inf):
                result += str(b)
            else:
                result += str(int(b))
            for var in range(len(row)-1):
                coeff = row[var]
                if coeff == 0:
                    continue
                if (var == depth):
                    continue
                result += " + " + str(int(-row[var])) + "*" + var_names[var]
            if (not no_min and var != len(row)-1):
                result += ", "
        if (not no_min and var != len(row)-1):
            result = result[:-2] #remove comma at end
        if (not no_min):
            result += ")"
        
        result += ":"
        if is_exact:
            result += " //EXACT PROJ"
        else:
            result += " //INEXACT PROJ"
        
        result += "\n"
        depth += 1
    
    #Loop body
    result += indentation*depth
    result += "PRINT("
    for var in var_names:
        result += f"{var}, "
    result = result [:-2] #remove comma
    result += ")"
    return result

def INT_FME(Ab: np.matrix):
    bounds_list = []
    while (Ab.shape[1] > 1):
        Ab_new, bounds_new = INT_FME_project(Ab)
        Ab = Ab_new
        bounds_list.append(bounds_new)
        
    bounds_list.reverse()
    #print(bounds_list)
    
    loops = generate_loop_nest(bounds_list);
        
    return loops    
    
    
 
if __name__ == '__main__':

    filename = "sample1.txt"
    out_fname = "sample1_output_loop.txt"
    
    write_output = True
    n = len(sys.argv)-1

    if (n==0):
        print("usage is ./python3 INT_FME.py <input file> <output file[optional]>")
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


    result = INT_FME(Ab)
    print(result)
    

    #Write result to file
    if (write_output):
        out_file = open(out_fname, "w+")
        
        out_file.write(result)
        out_file.close()
    
    #print()
    #input("Press enter to continue...")
