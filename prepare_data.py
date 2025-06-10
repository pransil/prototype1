# Prepare the data
# 1. Read in .csv file, one line at a time
# 2. Open the output file 
# 2. First line is column headers
# 3. All other lines are data
# 4. If the 'window_start' value in one line is different from that value in the previous line
#   by (1744792560000000000 - 1744792500000000000)
#       for the previous line:
#           remove the 'ticker' value
#           remove the 'window_start' value
#           write the line to the output file
# 5. else:
#       for the previous line:
#           remove the 'ticker' value
#           remove the 'window_start' value
#           write the line to the output file