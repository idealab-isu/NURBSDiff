<!-- master thesis draft -->
# Abstrct

# 1. Introduction

Object consturction is a hot topic in computer graphics/computer vision. In most cases, researchers would generate meshes from point clouds(as well as other auxiliary information, such as normals and occupencies). Some achieve a good result, but the process is time-consuming. It takes much time to train a pre-trained model, and the inference process is also time-consuming. Based on the fact, we would like to have a light-weighted but full-fledged method which can retreieve most of objects(watertight or non-watertight) swiftly and accurately. So instead we try to explore the area of nurbs, which is a mature technology in autoCAD. 
**explain nurbs**
nurbs stands for non-uniform rational B-spline. It can represent a wide range of curves and surfaces as figures shown below. **duck1 obj**
This object is generated using nurbs-python lib(ref). Its source code can be found on github link. **link** With the help of nurbs-python and examples in geomdl(ref), we are capable of generating nurbs surface by setting control points, weights, degree, and knot vector. **explain control points, weights, degree, and knot vector**
NURBS surface is actually a surface which can be parameterized, which means that the surface can be represented by a set of complex functions. In detail, the surface is controlled by its control points, its weight, its degree, and its knot vector. The control points controls the shape of the surface, while the weight controls how the degree surface is twisted when approaching the corresponding control point. The degree of the surface controls the degree of base functions. Acutally, the NURBS surface is determined by its underlying base functions and control points. Last but not least, the knot vector, a non-decreasing vector determines the shape of the surface. We'll discuss the details in the following sections. Indeed, we would like to use nurbs surface to represent object. Fortunately, there are some works which have already done this. [name of the author] propsed a new method in [name of  nurbs diff], in which they created a nurbs module for nurbs which use deep learning algorithm to optimize the paramerters of nurbs surface in order to construct the surface. We mainly focus on the fifth point proppsed in this paper, in which they recontruct the surface from point clouds with unsueprvised learning. In addition, they have disclosed their repo [link is here], which is a great help for us to understand the details of their work. However, we found that their code is somewhat incompete, they seems to miss the code with unsupervised learning. So we follow their work and try to implement the unsupervised learning part to test if their method can work.  Then we also run poco repo(ref) to see the result of their work. **explain poco a bit**. With the results from two different repos, we can compare the results and see if the method proposed in [name of  nurbs diff] can work. Then we analysed the result generated by these two repos. Finally, we conclude our work and propose some potential improvements on nurbs-diff method.

**insert some figures here**




# 2. Related Work

## 2.1. NURBS-diff

## 2.2. POCO
**check **
The paper proposed a method to reconstruct a mesh from a point cloud using implicit neural networks. There are several novel ideas in this paper. First, to address the scalability issues of encoding the isosurface of the whole object into a latent vector, they instead generate latent vectors on a coarse regular 3d grid and use them to answer occupancy queries. In this way, the latent vector has a weak connection with sampled input point cloud.  Second, to solve the problem of discretization tuning when relying on fixied patches, the authors conduct a learning-based interpolation on nearest neightbors using inferred weights. With these tweaks, the authors are able to reconstruct a mesh from a point cloud with a high accuracy and better details.


# 3. Method

## 3.1. principle of nurbs surface

## 3.2. **principle of nurbs-diff  main point**

## 3.3. **principle of poco**

# 4. Experiments

## 4.1. dataset

## 4.2. **nurbs-diff**

## 4.3. **poco**

# 5. result

## 5.1. **nurbs-diff**
## supervised
## unsupervised

## 5.2. **poco**

# 5. Conclusion

# 6. References

# 7. Appendix

# 8. Acknowledgements

# 9. Author Contributions

# 10. Conflict of Interest

# 11. Footnotes

# 12. Tables

# 13. Figures

# 14. Algorithms

# 15. Code

# 16. References

- wikipedia definition of nurbs ?
- nurbs diff
- poco
- nurbs book: the book provided by lizeth at early stage
- Advanced Animation and rendering techniques: the book provided by lizeth at late stage
- meshlab
- github repo used
- my github repo
- rhino7(subd code)
- blender(uv mapping)

# 17. Appendix