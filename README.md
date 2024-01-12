# Py3BR
Python Three(3) Body Recombination, a Python package for three-body recomination of atoms using classical trajectories. 

## Installation
I recommend installing Py3BR in a conda environment:
```python
conda create --name myenv python
conda activate myenv
```

To install in the environment, navigate to the root directory of the repository and run 
```python
pip install . 
```

## Usage
<p> See <code>example</code> folder for example usage of the program. The input file <code>input.py</code> contains the parameters for the calculation. <code>sim3.py</code> in the same folder calculates trajectories and outputs the results to a short and long format file. 

## Units
### Input
All input parameters are in atomic units, except for collision energy which is in Kelvin. 

### Output
The collision energy is reported in Kelvin, just as in the input. All other attributes are in atomic units. The opacity function is a unitless probability, the cross section is in $`\textrm{cm^5}`$, and the three-body recombination rate is in $`\textrm{cm^6/s}`$.

## License
 
[MIT](https://choosealicense.com/licenses/mit/)
