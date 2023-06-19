# README

# 1. How to use?

1. Build a folder with a specified PDE name. 

    Then create a new file 'xxx.ipynb'. When the 'xxx.ipynb' file has built in the child-folder, one should notice that the first several lines in the 'xxx.ipynb' file should be 

    ```Python
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath('__file__'))))

    from PINNS import *
    ```

    to import the files, classes in the parent folder.

2. Build the PINNS object.
    ```Python
    def u0(X: Tensor):
    x = X[:, 1]
    return th.sin(x).reshape((-1, 1))

    net = ANN((2, 1), (6, 6))
    pde = PDE_Square((0, 1), (0, 1), u0, 500)
    pinns = PINNS(net, pde)
    ```

    Here, one should be careful that the class 'PDE_Square' has no implement of the initial condition 'u0'. Thus, provided the u0, the pde will be built correctly.

3. Train and load.
   The 'ANN.py' has built-in methods for auto save and load. In the process of training, the folder './FOLDERNAME/models/' and './FOLDERNAME/models/autosave' will be built. And the model with the minimal loss will be save as 'best.pt' in the './FOLDERNAME/models' folder.

   When the net has the parameter 'loadBest', it will load the './FOLDERNAME/models/best.pt' as the model parameters. And one can also provide a non-empty content for the 'loadFile', then the 'ANN.py' will load the given path as the model.