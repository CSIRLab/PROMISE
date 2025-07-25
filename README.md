# PROMISim V1.0

The PROMISim framework was developed by [Prof. Ningyuan Cao's group](https://engineering.nd.edu/faculty/ningyuan-cao/) (University of Notre Dame). The model is made publicly available on a non-commercial basis. Copyright of the model is maintained by the developers.

:star2: This is the initial release (v1.0 under development) of the CIM tool. The current version focuses exclusively on modeling and simulation of probabilistic memory architectures implemented in bulk CMOS technology. A newly designed user-friendly and highly visual interactive interface has also been introduced to enhance usability and simulation experience.


## File lists
The simulator core is implemented in both Jupyter Notebook and standalone Python script versions, stored in separate directories. The Jupyter-based version supports modular and distributed execution, allowing users to better understand the internal mechanisms of the simulator. In contrast, the Python version is fully packaged and ready to run directly. Both versions share identical file structures and functionalities.
1. **Manual**
   `Documents/User Manual of Probabilistic Memory for Intelligent Systems at Edge (PROMISE) V1.pdf` 
   → User guide for understanding and operating the PROMISE simulator.

2. **Data** 
   `Data/` 
   → Contains custom RNG (random number generator) definitions and parameter files for various neural network models.

3. **Figures** 
   `fig/` 
   → Stores all plots and visualizations generated by the simulator.

4. **Interface Module** 
   `interface/` 
   → Contains all code related to the interactive user interface and parameter transmission between front-end and backend modules.

5. **Periphery Simulation Modules** 
   `periphery/` 
   → Includes simulations from device-level to circuit-level, covering memory cells, analog behavior, and peripheral components.

6. **High-level Architecture Simulator** 
   `simulator/` 
   → Contains high-level simulation logic for different memory architectures and compute paradigms based on probabilistic memory.


## Support for CIM hardware evaluations with technology scaling down to xxnm node.
```
Please specify the following parameters in the Param.cpp for CIM evaluation.

to be done

Please also set the parameter "technode" to specify the technology node (from 130nm to xxnm).

```

This research is supported by ....



## References related to this tool 

