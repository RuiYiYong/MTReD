""" 
This script computes the reprojection error for each Colmap output from points3D.txt files. 

"""


FILES = ["PATH TO A points3D.txt file", "PATH TO ANOTHER"]

# Loop files
for file in FILES:
    # Initialise variables
    total_error = 0
    count = 0
    
    # Compute reprojection error
    fp = open(file, 'r')
    line = True
    while line:
        line = fp.readline()
        if (len(line)):
            if (line[0] != '#'):
                count += 1 
                total_error += float(line.split(' ')[7])
    avg_reproj_error = total_error/count
    
    # You can compile errors as desired, here we just print a value
    print(f"Your average reprojection error is {avg_reproj_error}")