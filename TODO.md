# TODO

## Change code layout

The inclusion of jbLib as a subtree of the git repository confuses the make build because jbLib doesn't use the same make structure. I think the layout of the code needs changing to flatter structure, possibly:

	core/
	physics/
	solvers/
	monitors/
	
each with the header and source files together within the folder. Then the make file can be dramatically simplified. I think a lot of the auto-target finding is counter productive.