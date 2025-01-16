
def compute_hypervolume_2d(points, reference_point):
    """
    Compute the hypervolume for a set of 2D points.
    
    Parameters:
    - points: list of tuples, each representing a point in 2D space.
    - reference_point: tuple, the reference point (x_ref, y_ref).
    
    Returns:
    - hypervolume: float, the calculated hypervolume.
    """
    # Sort points by the first objective (x-coordinate) in descending order
    sorted_points = sorted(points, key=lambda p: p[0], reverse=True)
    
    hypervolume = 0.0
    prev_x = reference_point[0]
    
    for x, y in sorted_points:
        # Calculate the area of the rectangle
        width = prev_x - x
        height = reference_point[1] - y
        hypervolume += width * height
        prev_x = x  # Update for the next iteration
    
    return hypervolume

# Example usage
points_2d = [(0.8, 0.9), (0.6, 0.7), (0.5, 0.8)]
reference_2d = (1.0, 1.0)
hv_2d = compute_hypervolume_2d(points_2d, reference_2d)
print("2D Hypervolume:", hv_2d)



def compute_hypervolume_recursive(points, reference_point):
    """
    Compute the hypervolume for a set of multi-dimensional points recursively.
    
    Parameters:
    - points: list of tuples, each representing a point in N-dimensional space.
    - reference_point: tuple, the reference point in N-dimensional space.
    
    Returns:
    - hypervolume: float, the calculated hypervolume.
    """
    if len(reference_point) == 2:
        # Base case: 2D hypervolume
        return compute_hypervolume_2d(points, reference_point)
    
    # Sort points by the first objective in descending order
    sorted_points = sorted(points, key=lambda p: p[0], reverse=True)
    
    hypervolume = 0.0
    prev_x = reference_point[0]
    
    for point in sorted_points:
        width = prev_x - point[0]
        # Reduce dimensionality for recursion
        reduced_points = [p[1:] for p in sorted_points if p[0] >= point[0]]
        reduced_reference = reference_point[1:]
        height = compute_hypervolume_recursive(reduced_points, reduced_reference)
        hypervolume += width * height
        prev_x = point[0]
    
    return hypervolume

# Example usage
points_3d = [(0.8, 0.9, 0.7), (0.6, 0.8, 0.6), (0.7, 0.6, 0.9)]
reference_3d = (1.0, 1.0, 1.0)
hv_3d = compute_hypervolume_recursive(points_3d, reference_3d)
print("3D Hypervolume:", hv_3d)
