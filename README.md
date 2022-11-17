# irrigation_abm

### Summary

As access to water is a right of all people, government agents are responsible to allocate water to guarantee its sustainable use for multiple users
However, to decide the best allocation strategy is not a straightforward task, as in complex systems, which depend on a collection of individual decisions by people.
This is a hydrological model that take into account only water users for irrigation purposes.
There are two agents that interact with each other and the environment: Farmer Agent that asks for water rights and withdraws water from the canal,
and Manager Agent, that distribute water rights to farmers considering current water level in the canal. Farmers can override manager decision,
based on intrinsic properties and withdraws water anyways.
We assess the impact of oversight by restricting override capabilities from farmers (they are scared of being caught).

The model uses several Mesa concepts and features using real data:

- Network Grid (graph-based using networkx)
- Custom schedule
- Dynamically adding agents from the schedule
- Inheriting behavior based on data
- Interaction among agents and the environment

### How to run

We recommend installing anaconda, then installing mesa in the virtual environment.

In case trying to run this model, contact me for a sample version of water rights dataset

### Files

`Main.py`: defines the irrigation abm model itself

`plots.py`: contain codes for visualization

`scheduler.py`: Defines a custom variant on the StagedActivation scheduler

`data_preparation.py`: contains supporting functions to start the model such as: fit distributions, read .xlsx, etc

### Further reading

I intend to publish a paper with this current model. Stay tuned

For custom scheduler, please see the [wolf-sheep example](https://github.com/projectmesa/mesa/tree/main/examples/wolf_sheep)

For network-based abm, please see the [virus_on_network example](https://github.com/projectmesa/mesa/tree/main/examples/virus_on_network)

### Support

Fell free to contact me at any issue