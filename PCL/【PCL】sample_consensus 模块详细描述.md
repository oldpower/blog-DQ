## 【PCL】sample_consensus 模块详细描述

以下是支持的模块及其描述：

---

### 支持的模块

1. **SACMODEL_PLANE** 用于确定平面模型。平面的四个系数由其 Hessian 法线形式表示：`[normal_x normal_y normal_z d]`

2. **SACMODEL_LINE** 用于确定直线模型。直线的六个系数由直线上的一个点和直线的方向表示：`[point_on_line.x point_on_line.y point_on_line.z line_direction.x line_direction.y line_direction.z]`

3. **SACMODEL_CIRCLE2D** 用于确定平面中的 2D 圆。圆的三个系数由其中心和半径表示：`[center.x center.y radius]`

4. **SACMODEL_CIRCLE3D** 用于确定平面中的 3D 圆。圆的七个系数由其中心、半径和法线表示：`[center.x, center.y, center.z, radius, normal.x, normal.y, normal.z]`

5. **SACMODEL_SPHERE** 用于确定球体模型。球体的四个系数由其 3D 中心和半径表示：`[center.x center.y center.z radius]`

6. **SACMODEL_CYLINDER** 用于确定圆柱模型。圆柱的七个系数由其轴上的一个点、轴方向和半径表示：`[point_on_axis.x point_on_axis.y point_on_axis.z axis_direction.x axis_direction.y axis_direction.z radius]`

7. **SACMODEL_CONE** 用于确定圆锥模型。圆锥的七个系数由其顶点、轴方向和开口角度表示：`[apex.x, apex.y, apex.z, axis_direction.x, axis_direction.y, axis_direction.z, opening_angle]`

8. **SACMODEL_TORUS** 用于确定圆环模型。圆环的八个系数由其内半径、外半径、中心点和圆环轴表示。

9. **SACMODEL_PARALLEL_LINE** 用于确定与给定轴平行的直线模型，允许在最大指定角度偏差内。直线系数与 `SACMODEL_LINE` 类似。

10. **SACMODEL_PERPENDICULAR_PLANE** 用于确定与用户指定轴垂直的平面模型，允许在最大指定角度偏差内。平面系数与 `SACMODEL_PLANE` 类似。

11. **SACMODEL_PARALLEL_LINES** 尚未实现。

12. **SACMODEL_NORMAL_PLANE** 用于确定平面模型，并附加约束：每个内点的表面法线必须与输出平面的表面法线平行，允许在最大指定角度偏差内。平面系数与 `SACMODEL_PLANE` 类似。

13. **SACMODEL_NORMAL_SPHERE** 类似于 `SACMODEL_SPHERE`，但附加了表面法线约束。

14. **SACMODEL_PARALLEL_PLANE** 用于确定与用户指定轴平行的平面模型，允许在最大指定角度偏差内。平面系数与 `SACMODEL_PLANE` 类似。

15. **SACMODEL_NORMAL_PARALLEL_PLANE** 定义了一个用于 3D 平面分割的模型，使用附加的表面法线约束。平面法线必须与用户指定的轴平行。因此，`SACMODEL_NORMAL_PARALLEL_PLANE` 等同于 `SACMODEL_NORMAL_PLANE` + `SACMODEL_PERPENDICULAR_PLANE`。平面系数与 `SACMODEL_PLANE` 类似。

16. **SACMODEL_STICK** 用于 3D 棒状物分割的模型。棒状物是具有用户给定最小/最大宽度的直线。

17. **SACMODEL_ELLIPSE3D** 用于确定平面中的 3D 椭圆。椭圆的十一个系数由其中心、半轴和法线表示：`[center.x, center.y, center.z, semi_axis.u, semi_axis.v, normal.x, normal.y, normal.z, u.x, u.y, u.z]`

---

### 支持的鲁棒样本一致性估计器

1. **SAC_RANSAC** RANdom SAmple Consensus（随机采样一致性）。

2. **SAC_LMEDS** Least Median of Squares（最小中值平方）。

3. **SAC_MSAC** M-Estimator SAmple Consensus（M估计采样一致性）。

4. **SAC_RRANSAC** Randomized RANSAC（随机化 RANSAC）。

5. **SAC_RMSAC** Randomized MSAC（随机化 MSAC）。

6. **SAC_MLESAC** Maximum LikeLihood Estimation SAmple Consensus（最大似然估计采样一致性）。

7. **SAC_PROSAC** PROgressive SAmple Consensus（渐进采样一致性）。

---

### 默认建议
如果您对上述大多数估计器及其操作方式不熟悉，建议使用 **RANSAC** 来测试您的假设。

---

### 总结
`pcl_sample_consensus` 模块支持多种几何模型和鲁棒估计器，能够从点云数据中提取平面、直线、圆、球体、圆柱、圆锥等几何形状，并处理噪声和异常值。通过选择合适的模型和估计器，可以满足不同的应用需求。