# Report 1

## what is Nurbs Surface?
Nurbs Surface is composed of control points, knot vector as well as weights.

>Control points: Control points are 3D points which can be used to control the shape of the surface. Weights are 1D vector which assigns weight to each control point.

>Degree: The degree of a NURBS curve or surface determines the order of the basis functions that are used to construct the curve or surface. For example, a degree of 2 means that quadratic basis functions will be used to construct the curve or surface.

>Knot vector: The knot vector which divides the curve into segments, on the other hand, is a sequence of non-decreasing real numbers that define the parameterization of the B-spline curve. Knot points determines basic function(e.g. Cox-De Boor algorithm). The following equation shows the how the basic function is computed.
<center>N<sub>i, j</sub> = (t - U[i]) / (U[i+j] - U[i]) * N<sub>i, j-1</sub> + (U[i+j+1] - t)/(U[i+j+1] - U[i+1]) * N<sub>i+1, j-1</sub></center>

>The basic function can be computed recursively. This calculates the basis functions of degree j based on the basis functions of degree j-1.



>Additionally, the number of control points, the degree of the curve, and the knot vector are related by the equation:
<center>len(control points) + degree + 1 = len(knot vector)</center>


## How does NURBS-DIFF deal with knot_u, knot_v mapping of target surface?
In [examples/test/DuckyNURBSSurfaceFitting.py](../examples/test/DuckyNURBSSurfaceFitting.py), [knot_u of target surface](../examples/test/DuckyNURBSSurfaceFitting.py#L137) is the following:
```python
knot_u = [0.         0.         0.         0.         0.16666667 0.33333365
        0.5        0.66666635 0.75726286 0.83108925 0.83333333 0.8973262
        0.91289152 0.96520244 1.         1.         1.         1.        ]
 ```
Since len(num_ctrl_pts1) = 14, degree_u = 3, so the length of knot_u is 18, which fits the equation mentioned above.

The knot_u is passed to [SurfEval](../NURBSDiff/surf_eval.py#L16) class afterwards, which is a class for constructing target surface with regard to given knots and ctrpts. Here the original knot_u is transformed to a list with 512 items since [num_eval_pts_u](../examples/test/DuckyNURBSSurfaceFitting.py#L155) = 512. The transofrmed knot_u is displayed in the following:
```python
knot_u = [ 3, 3, 3..., 4, 4, 4..., 13, 13, 13]
```
I used Counter to transform it to a dict, the result is:

```python
knot_u_counter = Counter({3: 86, 4: 85, 5: 85, 6: 85, 7: 47, 8: 37, 10: 33, 12: 27, 13: 18, 11: 8, 9: 1})
```
It refers that 3 repeated 86 times, since 86 / 512 $\approx$ 0.167, it seems the transformed knot_u removes all boundary knots(0, 1) to become the following.
```python
knot_u = [0.16666667 0.33333365
        0.5        0.66666635 0.75726286 0.83108925 0.83333333 0.8973262
        0.91289152 0.96520244 1.]
```
And then it extends its size to num_eval_pts_u. The starting element is 3, which is the degree on u mapping. The repeated times is (knot_u[current] - knot_u[current - 1]) by num_eval_pts_u.

For instance, the second different element is degree_u + 1 = 4, and it repeated 512 * (0.33333365 - 0.16666667)  $\approx$ 85.

In summary, the knot_u of target surface follows the original distribution so the plotting ploted by matplotlib is simliart to which plotted by geomdl. Since the parameter diff is just scaling. Knot_v follows the same rule.
## How does NURBS-DIFF deal with knot_u, knot_v mapping of predicted surface?
For training, [knot_int_u](../examples/test/DuckyNURBSSurfaceFitting.py#L168) is initlized as:
```python
knot_int_u= torch.nn.Parameter(torch.ones(num_ctrl_pts1+p+1-2*p-1).unsqueeze(0).cuda(), requires_grad=True)
```
As we can see, knot_int_u is initlized to a uniform torch vector. Afterwards, it's passed to this [optimizer](../examples/test/DuckyNURBSSurfaceFitting.py#L176) for training.
```python
opt2 = torch.optim.SGD(iter([knot_int_u, knot_int_v]), lr=1e-3)
```
But it seems there exists some issues in [their training python file](../NURBSDiff/nurbs_eval.py), the final output knot_int_u is still an uniform vector. After transforming it to Counter. It becomes
```python
knot_u_counter = Counter({3: 47, 4: 47, 5: 47, 6: 47, 7: 47, 8: 46, 9: 46, 10: 46, 11: 46, 12: 46, 13: 46})
```
So currently the uv span mapping loses its original distribution and turns to be uniform. It may be fixed after I check the code carefully.
## Result of Plotting target and predicted surface with different color regarding u and v
I used three basic colors for plotting. Pure red means u=v=0. Pure green means u=v=0.5, while pure blue means u=v=1(assuming knot_u, knot_v are normalized)

Since I got [knot_u as well as knot_v](../examples/test/DuckyNURBSSurfaceFitting.py#L222-L223) from previous steps, I am capable to plot the surface blocks by blocks. Each block is the collection of the elements with same value. More deatails can be found in [plot_subfigure](../examples/test/DuckyNURBSSurfaceFitting.py#L=61). I also plotted the surface just after the first epoch. Besides, I plotted the diff surface by using (target surface - predicted surface).

Here's the result:

[ducky fitting without plotting control points](../examples/test/ducky_reparameterization_no_ctrpts.pdf)

[ducky fitting with control points](../examples/test/ducky_reparameterization.pdf)

[customized surface fitting without plotting control points](../examples/test/surface_reparameterization_no_ctrpts.pdf)

[customized surface fitting with control points](../examples/test/surface_reparameterization.pdf)

## The points which may imporve
1. fix? or make knot_u, knot_v changeable during training
2. use other loss functions rather than mle