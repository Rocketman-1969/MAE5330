import numpy as np # Imports the numpy library and creates the alias np
from IPython.display import display # Used to display variables nicely in Jupyter
from mav_sim.chap2.transforms import rot_x,rot_y,rot_z

A=np.array([[10,9],
            [8 ,7],
            [6 , 5]])

R=rot_x(np.pi/3)@rot_z(np.pi/4)
ans=R@A
display(ans)


