###### Visualization configs ################################################################
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, ListedColormap
colors = [(1, 1, 1),
          (0, 1, 0),  
          (1.0, 0, 0.0)]  # Color for 1 (red)

# Create a ListedColormap
incmap = ListedColormap(colors)

colors = [(0.863, 0.824, 0.80),  # Very light gray for 0
          (0.0, 0.0, 1.0),    # Blue for (0, 1]
          (0.098, 0.098, 0.439),    # Dark blue for (1, 2]
          (0.627, 0.125, 0.941),    # Violet for (2, 3]
          (0.294, 0, 0.51),    # Dark violet for (3, 4]
          (0.855, 0.439, 0.839)]    # Red for > 4
outcmap = ListedColormap(colors)
boundaries = [0, 4, 4.5, 5, 5.5, 7.0]
outnorm = BoundaryNorm(boundaries, outcmap.N, clip=True)