# path-toolkit

This package contains the class `Path` which is useful for computing geometrical properties.

## Instantiating a path

```python
from path_toolkit import Path

# Either from a list of points or (n, 2) numpy array
path1 = Path(points=[[0, 0], [1, 0], [2, 1]])

# Or from seperated x and y coordinates (numpy arrays are allowed as well) 
path2 = Path(x=[0, 1, 2], y=[0, 0, 1])
```

## Class properties

These properties are lazily evaluated, such that the desired property is computed on first access.
The result is cached such that no additional computation is required on repeated access.

| Property                | Description                                 |
|:------------------------|:--------------------------------------------|
| `points`                | The points of the path.                     |
| `x`                     | The x coordinates.                          |
| `y`                     | The y coordinates.                          |
| `length`                | The length of the entire path.              |
| `path_length_per_point` | The length of the path at each point.       |
| `orientation`           | The orientation of the path for each point. |
| `curvature`             | The curvature for each point.               |
| `unit_tangent_vector`   | The unit tangent vector at each point.      |

## Class functions

| Function                       | Description                                                                                                                                                          |
|:-------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `resampled_path`               | Returns the path with equidistantly resampled points.                                                                                                                |
| `smoothed_path`                | Returns the smoothed path using [this](https://autowarefoundation.github.io/autoware.universe/main/planning/autoware_path_smoother/docs/eb/) approach from Autoware. |
| `without_duplicate_points`     | Returns a path without consecutive duplicate points.                                                                                                                 |
| `find_circle_segments`         | Returns a list of circle segments with the starting end ending index of the point list and respective radius.                                                        |
| `compute_circle_fit_curvature` | Returns the curvature profile of the path using the circle segments from `find_circle_segments`.                                                                     |