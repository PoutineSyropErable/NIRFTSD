# Neural Implicit Reduced Fixed Topology Solid Deformation

The goal is to use neural network to speedup physics simulation of solid deformation.
Fixed topology means it would need to be trained for different shape.

The end project would have a pipeline of
(Physics simulation parameter) --(simulation encoder)--> Encoded simulations --(octree sdf calculator)--> Sdf queries values --(opencl code)--> screen space recreation.

It currently works with:

(Phys sim parameters) ---(expensive physics simulation)----> list of points cloud --(mesh encoding)--> List of latent vectors
"Comment" --(sdf calculator)--> list of sdfs values --(marching cube recreation)--> recreated shape
"Comment" --(whatever 3d engine is used)--> screen space recreation

Note: "comment is used to add newlines without wrapping. To be removed, just imagine it doesn't exist for now"
since -- is a special markdown symbol

---

# More Detail

So, it's a proof of concept, and the new neural networks needs to be written to use the data currently generated to skip some steps.

But for now, it's nowhere near that, a proof of concept of the pipeline has been written and is usable.
It needs some refactoring, and documentation, and updating misc stuff to be consistent with the code state, even if the code work.
To be done later.

Current Todo List:
Add fourier features between physics simulation data generation and mesh encoding pipeline, so we'd encode a
fourrier feature mapped of the point cloud, in a higher dimension.

Regenerate data for different head push position (Body force makes too minor a change), and remove redundants simulation.
Only have points on the head, with normal /toward center direction force.
No need for too "upward or downward", as it shouldn't change the head tilt that much.

Keep the position/direction -> simulation.

Then, Make it work with multiple simulaitons, by appending, changing on the batch size is setup (Same simulation one after another)
Then fine tune the cosine similairty penalty to work with ~ $||u_1 t - u_2 t||$ rather then just $$|t_2 - t_1|$$, so it can have context of
different simulations, as two different simulation pairs one should have different loss for similarity.

We'll have a trained mesh encoder and sdf calculator.
Then, replace the physics simulation -> Mesh encoder pipeline.
We can train an ,
Simulation argument (Force position, direction, norm) -> List of encoded mesh.  
Hence, $$\mathbb{R}^3 -> \mathbb{R}^{n * T_c}$$
Where n is the dimension of the reduced latent space, and T (t_capital) is the number of time indices (key frame in a simulation)

Once this is donce, we can replace the sdf calculator to use an octree structures, allowing for faster calculations.

Then, dump the weights of the mesh encoder and the sdf calculator into a readable file, and in (C++ And opencl (cuda equiv)), recreate whatever
magic nvidia has to have the sdf -> 2d screen space (pixels) recreation of the shape.

Skipping the
sdf --(marching cube or ray/sphere marching/whatever)--> 3d world space --(projection)--> 2d screen space.
Allowing for even more speed

---

# Old stuff, old readme

\*\* The readme might not be perfectly consitent with the state. (In particular, the todo list, as it might already be done)

All important files are on ./main
YOU SHOULD HAVE HIDDEN FILES (.name.ext) FILES VISIBLES.
THE READMES OF EACH DIR ARE ".\_\_readme.md"

[View the PDF Report](./Latex_Report/main.pdf)
It's located at ./Latex_Report/main.pdfI

In main, the files are written in logical order, of writting and using them.
They are named in the order that you'd use them.
There's even a bash file that runs everything for you.

Link to the google docs with the files that aren't git tracked
https://drive.google.com/drive/folders/1vpslQHkE8iSuIB2wGIodClNLWq8q9kgH?usp=sharing
**Check the animations in CHECK_ME**

On the google drive, there is only the first 30 sdf rather then 2000 of them. I'm aiming for 550 on my machine, which will be like 200ish GB of
data. Could be fine tuned. But No time for that.

Anyway, the AI training will be more about potentially showing it could be done. Completing the training or fine tuning the pipe line ish
not something i have time to do.

Maybe in my free time, I'll do it later.

---

Todo List:

13_incremental_train_model.py # Aiming to finish this one tomorrow
14_use_model_for_calculation.py # Optional, to test the data
15_recreate_mesh_from_model.py # Optional, practice, a single mesh.
16_physics_simulation_from_model.py # Takes Finger position and shows the physics animation deformation
17_calculate_model_errors.py # Would be nice for scientific purposes, but there's no time I'll time to do that one.
// Just the run time itself will take too long.
