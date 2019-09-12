Overview: <br>
--------------------------------------------------------------------------------------
<div>In general neural networks have a fixed structure. Is this always necessary? It is known that the more deeper you go into the network, the more sophisticated features it learns.
After a certain layer, the network might be confident in learning on what the image is but because of having a fixed structure always, it has to use the entire layer to do all the computations and thus affecting the performance.
Can we construct a model that decides on the go whether or not a layer to be executed depending on what it learns and at the same time give the performance we need? 
The idea is that we do not need all the layers in determining the class for the object. A network ConvNet-AIG (Convolutional Networks with Adaptive Inference Graphs) with a gating unit is introduced that explain their network topology particularly accustomed on the input image. 
One way to do it is the use of Gumbel Max distribution that helps in making a discrete decision based on the relevance score. Another intuitive way is to incorporate the gating choice based on a policy through the concept of reinforcement learning. 
<br> </div>

Architecture Overviews: <br>
--------------------------------------------------------------------------------------
![Overall Architecture](/images/overall-archi.png) <br>
![Overall Architecture 1](/images/overall-archi1.png) 
<br>
<div align="center"> Figure: General architecture for the implementation. <br> 
(Top L-R): Structures for feed forward CNN, ResNet and Adanet with gating unit. <br>
Gating unit determines whether the layer needs to be executed for the current input image. <br> </div>

Existing Approach: <br>
--------------------------------------------------------------------------------------
![Existing Approach](/images/existing_app.png) <br>
<div align="center">  Figure: Gating unit in the basic block of resnet that determines whether or not to execute the layer. </div> 

Current Approach: <br>
--------------------------------------------------------------------------------------
![Current Approach](/images/current_app.png) <br>
<div align="center">  Figure: Current architecture with reinforcement learning. <br>
Implementing a policy that will help in making the discrete decision.  
 </div> <br>

Image Ref: <br>
--------------------------------------------------------------------------------------
Veit, A., & Belongie, S. (2018). Convolutional networks with adaptive inference graphs. In Proceedings of the European Conference on Computer Vision (ECCV) (pp. 3-18) </br>

Citing <br>
--------------------------------------------------------------------------------------
@conference{Veit2018, <br>
title = {Convolutional Networks with Adaptive Inference Graphs}, <br>
author = {Andreas Veit and Serge Belongie}, <br>
year = {2018}, <br>
journal = {European Conference on Computer Vision (ECCV)}, <br>
}
