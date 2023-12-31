# Highest Above Nearest Drainage Calculation

This code solves for the Highest Above Nearest Drainage (HAND) values of cells from a Digital Elevation Model (DEM). Data processing is done in two major stems:

1. Searching for all drainage connected to a node.
2. Selecting the drainage node where the HAND value will be derived from for a given starting node.

The HAND values are the differences in the elevation of the starting node and the selected node.

## Input Data
The input data required is the network data in JSON format. With the following sample content:

```json
{
    "0": {
        "row": 0,
        "col": 0,
        "accum": 1.3729063722245636,
        "elev": 650.0,
        "neighbors": [[1, 1], [0, 1]]
    }, 
    "1": {
        "row": 0,
        "col": 1,
        "accum": 3.609570510679968,
        "elev": 641.0,
        "neighbors": [[1, 2], [0, 2]]
    },
    "2": {
        "row": 0,
        "col": 2,
        "accum": 7.949967548702894,
        "elev": 629.0, 
        "neighbors": [[1, 3], [0, 3]] 
        }
    //-- snip --
}
```
The keys represent the node id which is generated from the rows and columns of the current node with a simple function:

```python
# node_id - the resulting single number id
# node_row - the row of the node in the DEM
# node_col - the column of the node in the DEM
# dem_cols - the total number of columns in the DEM
node_id = node_row * dem_cols + node_col
```

The data must be arrange in order of increasing node_ids (nodes going left to right from the topmost row and repeat the same pattern in the next lower rows).

The **neighbors** field contains the rows and and columns of the neighbors or specifically the cells identified during the flow accumulation which receives flow the current node.

The accumulation level and elevations are contained in the aptly named fields.

## Algorithm Inputs
Running the program with:
```cmd
hand_calculate --help
```
provides the following help text:
```cmd
Calculates the Heighest Above Nearest Drainage (HAND) values give the network data. The network data is a json file contaning the node_id incrementing from 0 to the data length. Each node must contain the accumulation value, row and column location of the node,the flow accumulation value and the list of neighbor nodes.

Options:
  -i, --input-file  input file name
  -r, --rows        number of DEM rows
  -c, --cols        number of DEM columns
  -t, --drainage-threshold
                    minimum accumulation ammount to clasify a node as drainage
  -d, --max-drainage
                    maximum number of connected drainage (default = 5)
  -l, --max-path-length
                    maximum number of connected drainage (default = 30)
  -a, --alpha       path length bias over average accumulation value (range 0 to 1) (default = 0.9)
  -p, --plot        plot the result or not
  --help            display usage information
```

Make sure that the **hand_calculate.exe** is in the root folder together with the **accumulations** folder. Running the executable for example can be done as,

```cmd
hand_calculate -i "accumulations\cells.json" -r 1047 -c 1613 -t 1000 -d 5 -l 40 -a 0.9 -p
```

The output files are the HAND numpy array,a JSON file containing the connected neighbors per node_id and another JSON file where each node is paired to the selected drainage node.

## Algorithms Used
### Searching for Connected Drainage
The **search_drainage** function uses Breadth-First Search algorithm to find the closest drainage to each starting node. The search is slightly modified to prevent exploration from drainage cells (as identified using the drainage threshold value) to its co-drainage cells.

The search returns multiple values if the flow-accumulation algorithm used a Multiple-Flow-Direction (MFD) approach. The user provided **max_drainage** value limits the number of connected drainage nodes. Nodes are sorted from least to greatest *Manhattan Distance* from the starting node where,

$${\text{manhattan}(\text{node}_A,\text{node}_B)} = \left|(\text{node}_A.\text{row} - \text{node}_B.\text{row})\right| + \left|(\text{node}_A.\text{col} - \text{node}_B.\text{col})\right|$$

### Finding Paths from a Node to a Drainage
The **find_all_paths** function is a wrapper for the **dfs_recursive** function which returns all possible paths from a starting node to a drainage node. The latter function uses a recursive approach while employing Depth-First Search (DFS) traversal. During the recursion, a copy of the path-so-far is being kept and a depth first search is recursively apply to extend the path until a drainage is found. A list of explored nodes is referenced to prevent a possible possible cyclic paths.

To improve searching speed, the user provied **max_path_length** parameter terminates the search if the current path length is over the said value. This is okay for nodes that have multiple paths leading up into a drainage but this would also affect nodes that are further away from drainage nodes (areas in relatively higher elevation with large slopes) where no path is going to be returned. This however, would not affect the resulting HAND calculations that much since high risk areas are generally closer to streams.

### Selecting the Final Path
Once all of the possible paths for all connected drainage nodes are identified, each will be ranked according to a certain score controlled by the **alpha** $(\alpha)$. This is a range of value from 0 to 1 which controls the path length bias over the average accumulation in the path. This way, the user can adjust which factor dictates the final HAND value for node more. The paths are score as follows:

<!-- $$
\text{LS}_j = 1 - \frac{l_j}{\sum_{i=0}^n l_i}
$$ -->
<p align="center">
<img src="https://latex.codecogs.com/svg.image?\color{White}%5Ctext%7BLS%7D_j%20%3D%201%20-%20%5Cfrac%7Bl_j%7D%7B%5Csum_%7Bi%3D0%7D%5En%20l_i%7D%0D" />
</p>


<!-- $$
\text{AS}_j = \frac{\bar{a}_j}{\sum_{i=0}^n \bar{a}_i}
$$ -->
<p align="center">
<img src="https://latex.codecogs.com/svg.image?\color{White}%5Ctext%7BAS%7D_j%20%3D%20%5Cfrac%7B%5Cbar%7Ba%7D_j%7D%7B%5Csum_%7Bi%3D0%7D%5En%20%5Cbar%7Ba%7D_i%7D%0D" />
</p>


<!-- $$
\text{PS}_j = \alpha \cdot \text{LS}_j + (1-\alpha) \cdot \text{AS}_j
$$ -->
<p align="center">
<img src="https://latex.codecogs.com/svg.image?\color{White}%5Ctext%7BPS%7D_j%20%3D%20%5Calpha%20%5Ccdot%20%5Ctext%7BLS%7D_j%20%2B%20(1-%5Calpha)%20%5Ccdot%20%5Ctext%7BAS%7D_j%0D" />
</p>

Where,

<!-- $$
\begin{align*}
l_j &= \text{length of path } j\\
\bar{a}_j &= \text{average accumulation along path } j\\
\text{LS}_j &= \text{path length score}\\
\text{AS}_j &= \text{average accumulation score}\\
\text{PS}_j &= \text{path final score}
\end{align*}
$$ -->
<p align="center">
<img src="https://latex.codecogs.com/svg.image?\color{White}%5Cbegin%7Balign*%7D%0D%0Al_j%20%26%3D%20%5Ctext%7Blength%20of%20path%20%7D%20j%5C%5C%0D%0A%5Cbar%7Ba%7D_j%20%26%3D%20%5Ctext%7Baverage%20accumulation%20along%20path%20%7D%20j%5C%5C%0D%0A%5Ctext%7BLS%7D_j%20%26%3D%20%5Ctext%7Bpath%20length%20score%7D%5C%5C%0D%0A%5Ctext%7BAS%7D_j%20%26%3D%20%5Ctext%7Baverage%20accumulation%20score%7D%5C%5C%0D%0A%5Ctext%7BPS%7D_j%20%26%3D%20%5Ctext%7Bpath%20final%20score%7D%0D%0A%5Cend%7Balign*%7D%0D" />
</p>

To select the drainage node for the current node, we let,
$$f(n_i) = f: n_i \mapsto \text{PS}_i$$
$$g(n_i) = g: n_i \mapsto e_i$$
<!-- $$
P_j = \{\text{PS}_0,\text{PS}_1,...\text{PS}_k\}
$$ -->
<p align="center">
<img src="https://latex.codecogs.com/svg.image?\color{White}P_j%20%3D%20%5C%7B%5Ctext%7BPS%7D_0%2C%5Ctext%7BPS%7D_1%2C...%5Ctext%7BPS%7D_k%5C%7D%0D" />
</p>

<!-- $$
T = \{ n_0,n_1...n_i \in N| f(n_0) = f(n_1) \cdots = f(n_j) = \max(P_j)\}
$$ -->
<p align="center">
<img src="https://latex.codecogs.com/svg.image?\color{White}T%20%3D%20%5C%7B%20n_0%2Cn_1...n_i%20%5Cin%20N%7C%20f(n_0)%20%3D%20f(n_1)%20%5Ccdots%20%3D%20f(n_j)%20%3D%20%5Cmax(P_j)%5C%7D%0D" /></p>

The final node selection will be given as,
$$n_f = \text{arg max}(n_i) {f(n_i)}$$

In case of a tie the final node will be chosen as,
$$n_j^* = \text{arg max} (n_i) \{n_i \in T: g(n_i)\}$$

Where,
<!-- $$
\begin{align*}
    f(n_i) &= \text{Function mapping the node $n_i$ to its path score PS$_i$}\\
    g(n_i) &= \text{Function mapping the node $n_i$ to its elevation $e_i$}\\
    P_j &= \text{Scores of paths from starting node $n_j$ to all its connected drainage nodes}\\
    k &= \text{number of connected nodes to $n_j$}\\
    T &= \text{the set of nodes with scores equal to the max score $\max(P_j)$}\\
    n_j^* &= \text{the selected drainage for $n_j$}
\end{align*}
$$ -->
<p align="center">
<img src="https://latex.codecogs.com/svg.image?\color{White}%5Cbegin%7Balign*%7D%0D%0A%20%20%20%20f(n_i)%20%26%3D%20%5Ctext%7BFunction%20mapping%20the%20node%20%24n_i%24%20to%20its%20path%20score%20PS%24_i%24%7D%5C%5C%0D%0A%20%20%20%20g(n_i)%20%26%3D%20%5Ctext%7BFunction%20mapping%20the%20node%20%24n_i%24%20to%20its%20elevation%20%24e_i%24%7D%5C%5C%0D%0A%20%20%20%20P_j%20%26%3D%20%5Ctext%7BScores%20of%20paths%20from%20starting%20node%20%24n_j%24%20to%20all%20its%20connected%20drainage%20nodes%7D%5C%5C%0D%0A%20%20%20%20k%20%26%3D%20%5Ctext%7Bnumber%20of%20connected%20nodes%20to%20%24n_j%24%7D%5C%5C%0D%0A%20%20%20%20T%20%26%3D%20%5Ctext%7Bthe%20set%20of%20nodes%20with%20scores%20equal%20to%20the%20max%20score%20%24%5Cmax(P_j)%24%7D%5C%5C%0D%0A%20%20%20%20n_j%5E*%20%26%3D%20%5Ctext%7Bthe%20selected%20drainage%20for%20%24n_j%24%7D%0D%0A%5Cend%7Balign*%7D%0D" />
</p>

This all boils down with the highest path score and in case of tie, choose the neighbor with the higher elevation.

### Calculating the HAND Values
Once the drainage node $n_j^*$ for a node $n_j$ is determined the associated HAND value $h_j$ is then calculated as
<!-- $$
h_j = \begin{cases}
    g(n_j) - g(n_j^*), & \text{if\hspace{1em}} g(n_j) - g(n_j^*) \geq 0\\
    0, & \text{if\hspace{1em}} g(n_j) - g(n_j^*) < 0
\end{cases}
$$ -->
<p align="center">
<img src="https://latex.codecogs.com/svg.image?\color{White}h_j%20%3D%20%5Cbegin%7Bcases%7D%0D%0A%20%20%20%20g(n_j)%20-%20g(n_j%5E*)%2C%20%26%20%5Ctext%7Bif%5Chspace%7B1em%7D%7D%20g(n_j)%20-%20g(n_j%5E*)%20%5Cgeq%200%5C%5C%0D%0A%20%20%20%200%2C%20%26%20%5Ctext%7Bif%5Chspace%7B1em%7D%7D%20g(n_j)%20-%20g(n_j%5E*)%20%3C%200%0D%0A%5Cend%7Bcases%7D%0D" /></p>

## Optimizations Done
Both drainage search and path finding algorithms were implemented via multithreading to improve runtime speed. This result into magnitudes of improvements over a single threaded operation. Proper data structures were also selected such as employing hashsets to check whether or not a node is already explored. Double ended queues were used for operations that pops elements from the front of a vector. Some vectors were also pre-allocatted at some estimated capacity.


