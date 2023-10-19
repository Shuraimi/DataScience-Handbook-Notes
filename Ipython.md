## Intro
IPython stands for Interactive Python also called as Interactive Control Panel 

## Shell or Notebook
There are primarily 2 ways of using IPython:-
- IPython shell
- IPython notebook also called Jupyter Notebook
### Launching IPython Shell
We can launch the IPython shell by typing ==ipython== in command prompt.
### Launching Jupyter Notebook 
We can launch the Jupyter Notebook by typing Jupyter Notebook in the terminal.

## IPython 's tools to access help and documentation 


There are namely:-
1. ? to explore documentation
2. ?? to explore source code
3. TAB for autocompletion
### 1. ? to explore documentation

This ? character is used to explore documentation and other relevant information of object, object methods, methods or even functions(including user-defined functions which have docstring) and is a shorthand(to help() in python) to access this information 

`L? #to get documentation of object`
`len? #to get documentation of len function `
`L.insert? #to get documentation of the insert method in list`

### 2. ?? to access source code

`square??`
This gives the code to this function named square.

Therefore using ?/?? gives us a useful way to check what a function or module does.

### 3. TAB for autocompletion 
Refer image


## Keyboard shortcuts for IPython 
Will come with practise 

### IPython magic🪄 commands 

Magic commands are prefixed by % character.

These are designed to succinctly solve various common problems in data analysis.
These come in two flavours:-

1. Line magics :- denoted by single % and operate on single line of input 
2. Cell magics :- denoted by double % and operate on multiple lines of input.

%paste :- When you copy and paste a block of code from a website, the indentation and function markers give error when we paste. Therefore we use this command to overcome this error.
This command enters and executes the code.

Syntax:- write %paste and the paste the code below this command 

%cpaste :- This command opens up an interactive multiline prompt where you can paste 1/more blocks of code to execute in a batch.

running extern code %run
As u advance, you'll find yourself working in both IPython interpreter and as execute code in a code editor.
Instead of opening a new window, we can run the code in the same IPython session by the following code

`%run #followed byfilename`
 Functions defined within this file can also be used in your IPython session.

Timing code execution:- %timeit
This magic command is used to automatically measure the time taken for execution for a single line of code.

Syntax:- 
`%timeit followed by single line of code`

Benefit of %timeit is that it runs multiple runs for single line code to give more robust results.
Use of another % before %timeit turns it into a cell magic 🪄 to measure time for multiple lines of code.

### Help on magic functions 

To access documentation of any magic function, type 
`magicfunction?`

For example :- `%timeit?`

`%magic` is used to get general description of the available magic functions.

`%lsmagic` to get list of all the magic functions 

## Input and Output history 

### Python's IN and OUT objects
`In [1]: import math`
`In 2: math.sin(2)`
`Out[2]: 0.9092974268256817`
`In [3]: math.cos(2)`
`Out[3]: -0.4161468365471424`


The IN/OUT labels are are the variables created by Python and are automatically updated to reflect their this history.

`print(IN)`
The IN object is a list which keeps track of commands in order.
IN[1] refers to the first command in input.

Whereas the OUT object is not a list by a dictionary mapping inputs to outputs.
Note that not all inputs have outputs.
For example, print doesn't return anything i.e returns NONE and does not have output.

### Underscore Shortcuts and Previous Outputs
We can refer to the previous output using a single Underscore which means to give the last output because single Underscore is updated with the last output.
This is available in both Python shell as well as in IPython.
`print(_)`

We can access the second to last output using double underscore (available in IPython)
`print(__)`

Similarly to access the third to last output, skipping any commands with no output.
`print(___)`

More than three underscores are difficult for IPython to understand therefore we use OUT[X] or OUT_X to acces the output of line 'X'.

### Suppressing outputs

We sometimes don't want to store the output but just want to display the result. For this, we can terminate the statement with a semicolon.

`math.sin[0]+math.sin[4];`

Here the output is completed silently and neither does it get stored in output dictionary nor is displayed.

To access set of inputs,
`%history -n 1-4`
Gives a list of inputs.

## IPython and shell commands
Any code after the ! mark is run by system command line and not by python kernel.

### Quick introduction to shell 
Shell is a way to interact with the computer through commands.

### Shell commands in IPython 
Any commands that can be written
