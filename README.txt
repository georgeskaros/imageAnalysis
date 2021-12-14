In order to  run the aplication it is requared to pass some arguments with it 

#=======#
# TRAIN #
#=======#

"-t", "--train" 	is used as a switch to change the image colorizer to training mode 
"--training-set" 	is used to set the folder containing the training images 

example for training: 	python3 ImageColorizer.py -t --training-set images/train/

#==========#
# Colorize #
#==========#

"-c","--colorize"	is used as a switch to change the image colorizer to colorizing_image mode 
"-i","--input_image"	is used to select the image that has to be colorized 
"-o","--output_image"	is used to chuse an output file for thr colorized image
"--use-graphcuts"	is used as a graph cut algorithm switch

example for colorization:	python3 ImageColorizer.py -c -i images/test/orange.jpg -o images/results/colorized_without_graphcuts.jpg