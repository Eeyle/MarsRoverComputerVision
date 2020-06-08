# Is the Mars Rover taking a selfie?
## Using distributed machine learning with Spark for computer vision of the Mars Curiosity Rover
#### Scott Armstrong
#### Capstone Project for GA DSI 2020, 9 Jun 2020

### Introduction
As a government-funded enterprise the Mars Curiosity Rover takes images which are publicly available. It is of particular importance to use computer vision on these images due the sheer number of images it takes on a daily basis, as well as because the camera cannot be controlled in real-time. Over 6000 images were retrieved from the [Mars surface image labeled data set](https://data.nasa.gov/Space-Science/Mars-surface-image-Curiosity-rover-labeled-data-se/cjex-ucks)
compiled by Alice Stanboli and Kiri L. Wagstaff (DOI: 10.5281/zenodo.1049137, paper discussing the pictures: Kiri L. Wagstaff, You Lu, Alice Stanboli, Kevin Grimes, Thamme Gowda, and Jordan Padams. "Deep Mars: CNN Classification of Mars Imagery for the PDS Imaging Atlas." Proceedings of the Thirtieth Annual Conference on Innovative Applications of Artificial Intelligence, 2018.)
hosted on the [NASA Open Data Portal](https://nasa.github.io/data-nasa-gov-frontpage/). 

### Dataset
Data was provided as a set of ~6000 images of size 256x256 in full-color, but due to memory and time considerations the data was further compressed to 64x64 greyscale, though the distributed approach is more than capable of handling larger images. Images were labeled with one of twenty-four different categories representing different parts of the rover that the camera might be looking at. These categories were reduced to two for simplicity as it was not the focus of the project. Since repairs cannot be made to the rover yet, engineers and scientists require the camera to frequently point down at the rover itself in order to monitor deterioration, dust, and diagnose problems. Since the Martian surface is rather barren, the images ended up being split roughly evenly between pictures of the horizon/ground and pictures of the rover itself. _Is it possible to use computer vision to determine whether or not the Mars Curiosity Rover is taking a selfie?_

### Running
The project is a Scala 2.11 application built using sbt for Spark 2.4.5. The build configuration are found in *build.sbt* while source code is found in *src/*. Once cloned the project can be by submitting the application to a running master via `spark/bin/spark-submit --class Capstone --master <master> target/scala-2.11/capstone_2.11-0.1.jar`, provided the proper IP address is entered into the spark session declaration on line 32 of src/main/scala/Capstone.scala.

### Distribution
The project is structured in a way to easily allow distributed computing. A single line can be edited to be able to sumbit the application to a Spark Master running on a main computer. The project directory (along with a Spark distribution) can then be copied to as many other computers as are available, where each computer connects to the master and trains the model in parallel as a Spark Worker. This project is a proof-of-concept that home computer enthusiasts can distribute otherwise daunting tasks across several computers. Although the images included are only 64x64, and the dataset only includes about 6000 pictures, this approach is robust to scaling the data up a large amount with some proper data augmentation and larger, more colorful images.

### Processing
Image processing including downsizing and converting were done in Python for simplicity. The train, validation, and test sets of images were compiled into single csv files for easy reading by Spark. These csv files are the only data needed to be distributed among different computers, though for very large datasets a proper distributed file management system would be necessary.

### Results
Several models were tried but only one gave any major success. A random forest model resulted in a testing accuracy of 75%, with a maximum tree-depth of 16 nodes and 256 different trees in total. Computer vision can absolutely determine whether or not the Mars rover is taking a selfie, even when using small greyscale images.

### Future
This approach is robust in everything except disk space, so the largest datasets would still require a dedicated distributed file manager. This method works best for models that are capable of parallel training and for gridsearching large hyperparameter spaces. 

The most important next step is to include color, as that is the most notable difference between the rover and the landscape around it. The project is set up to scale perfectly well, so adding colors would only require a small change to the processing scripts and a bit more processor time distributed among computers. Then, images can be scaled to a more reasonable size, which will be necessary for distinguishing between the smaller machine parts of the rover which are labeled. Finally, these larger images would be ready for all kinds of augmentation, though this is the step that would require the largest jump in disk space.

Finally, once the images are larger and more colorful, if there is still room in the burning-hot processors of my three computers then the hyperparameters can be cross-validated for the final push to absolute accuracy.





