""" 
This script computes the average reprojection error for Colmap outputs using the points3D.txt 
output files. The script takes a list of paths to different points3D.txt files as an inputs and prints the 
result to the console for each file. 
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