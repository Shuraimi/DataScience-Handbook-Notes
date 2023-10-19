# Introduction to Numpy
Numpy in short status for Numerical Python which provides an efficient interface for storage and operation on data buffers.

To check the version of Numpy 
`numpy.__version__`

To import numpy
`import numpy as np`

To display all the contents of the numpy namespace 
`np.<TAB>`

For numpy documentation 
`np?`

## Understanding datatypes in python

The difference between C and Java and little is that the variables in C or Java are statically typed i.e we need to specify the data types of variables whereas in python the variables are fundamentally types meaning we can initialise variables without datatypes.

### A python integer in more than just an integer
For example, when we define an integer as 'x=100', x is not just a raw integer. It is a pointer to compound C structure which contains several values.

A single integer in Python 3.4 actually contains four pieces:
- ob_refcnt, a reference count that helps Python silently handle memory allocation and deallocation
- ob_type, which encodes the type of the variable
- ob_size, which specifies the size of the following data members
- ob_digit, which contains the actual integer value that we expect the Python variable to represent

![[Screenshot_2023-10-12-11-50-41-124-edit_com.foxit.mobile.pdf.lite.jpg]]

Here PyObject_HEAD is a part of the structure which contains the above mentioned parts.

The difference here is :-
A C integer is a label for memory postion where bytes encode an integer.
But python integer is a pointer to postion in memory containing all python object information including bytes that convert into integer.

### A python list more than just a list 
Lists are mutable and can stire heterogeneous datatypes but this  comes at a cost to offer flexibility, each item in a list must contain its type info, reference count and other info. Therefore each item is an object.

![[Screenshot_2023-10-12-12-08-29-479_com.foxit.mobile.pdf.lite.jpg]]

At the implementation level, the array essentially contains a single pointer to one contiguous block of data. The Python list, on the other hand, contains a pointer to a block of pointers, each of which in turn points to a full Python object like the Python integer we saw earlier. 

### Fixed type arrays in python 
We can create fixed for arrays using the arrays module but ndarray of numpy offers much more efficient operations.

### Creating arrays from python lists
We can use np.array ton create arrays from list.
`np.array([1,2,3,4])`

These should be of the same datatype or else they are upcast if possible
`np.array([3.13,1,2,4])`
Here the array is upcast to float .

To specify the upcast datatype or to explicitly specify the upcast dtype:-
`np.array([3.13,1,2,4], dtype='float32')`

### Creating arrays from scratch
For creating large arrays, it's helpful to use numpy built in routines.

Here are examples:-

To create an array of of length 10 integr type filled with zeroes
	`np.zeroes(10,dtype=int)`

Create a 3x5 matrix filled with 1's of floating dtype 
	`np.ones((3,5),dtype=float)`
	
Create a 3x5 array filled with 4
	`np.fill((3,5),4)`

Create an array filled with a linear sequence
Starting at 0, ending at 20, stepping by 2
(this is similar to the built-in range() function)
	`np.arange(0,20,2)`

Create an array with five values evenly spaced between 0 and 1
	`np.linspace(0,1,5)`

Create a 3x3 array of uniformly distributed random values between 0 and 1
	`np.random.random(3,3)`

Create a 3x3 array of normally distributed random values with mean 0 and standard deviation 1
	`np.random.normal(0,1,(3,3))`

Create a 3x3 array of random integers in the interval [0, 10]
	`np.random.randint(0,10,(3,3))`

Create a 3x3 identity matrix
	`np.eye(3)`

Create an uninitialized array of three integers
The values will be whatever happens to already exist at that memory location
	`np.empty(3)`

### Numpy standard dtypes
NumPy is built in C, the types will be familiar to users of C, Fortran, and other related languages.

Note that when constructing an array, you can specify them using a string:
`np.zeros(10, dtype='int16')`

Or using the associated NumPy object:
`np.zeros(10, dtype=np.int16)`

![[Screenshot_2023-10-12-12-46-45-203-edit_com.foxit.mobile.pdf.lite.jpg]]

Completed on 12/10/23
## The basics of Numpy arrays

Numpy array manipulation 
1. Attributes of arrays such as size, shape, memory size and data types of arrays
2. Indexing 
3. Slicing 
4. Reshaping
5. Joining 

First we'll create three arrays using the code 
`import numpy as np`

`np.random.seed(0)`
Seed is used for reproducibility i.e after this code all generated random arrays will have same numbers.

`x=np.random.randint(10,size=(3,4,5))`

Each array has attributes:-

`x.ndim`
`x.shape`
`x.size`
`x.dtype`

.itemsize lists the memory size in bytes of the each element in the list and .nbytes gives the total memory size of the list.

`x.itemsize`
`x.nbytes`

### Indexing
Indexing in Numpy is similar to indexing in python lists.

`x1[0]`
Gives the first element in the numpy array

To access from the end of the array, we use negative indexing 
`x1[-1]`

To access multi dimensional array, we use a comma separated tuple,
`x1[1,2]`

This means second row, third column since first row starts from 0.

Note:- In case you insert a floating point number into a int numpy array, the decimal places are truncated.

### Array slicing 
Slicing in numpy is similar to that in python which follows the syntax
`x[start:stop:step]`

By default, start=0
stop=dimension of the array
step=1

Examples:-

`x1[:5]`
All elements from 0(inclusive) to 5 (exclusive)

`x1[5:]`
All elements from 5 till the end

`x1[::2]`
All elements of array with step 2

Reverse every element of the array 
`x1[::-1]`


#### Multi dimensional arrays
The slicing can be done in the same way but only difference is that the it is separated by commas for row and columns.

Define a multi dimensional array 
`x2`

`x2[:2,:4]`
This means first 2 rows and first 3 columns.

And so on.

#### Accessing a row or column from array
It is often needed to access a single row or column by using the syntax.
This is done by combining slicing and Indexing.

To get the first column of array x
`x[:,0]`

Similarly to get the second row of x
`x[2,:]`
Or another syntax is 
`x[2]`


#### Subarrays as no-copy views
When we slice an array, it will give a view instead of copies of the array whereas in python lists, we get copies when we slice it.

`x3`
A mxn dimensional array 

Now we take a subarray from this array through slicing.
`x_subarr=x3[:2,:3]`

Now if we modify an element in this array, the original array element is also changed.
`x_subarr[0,0]=8`

`x3`
Print the array

#### Creating copies of arrays 
We use the `.copy()` to create a copy of the array.

`x_subarr=x3[:2,:3].copy()`

Now if we change any element of this x_subarr, the orginal array remains unchanged.

#### Reshaping arrays
We reshape the array using the `.reshape()`

`x4=np.arange(1,10).reshape((3,3))`

Note that for this to work, the size of initial array must be equal to the size of the reshaped array.

Another use is conversion of 1D array into 2D row or column vector.
`x=np.array([1,2,3])`

To make it into a row vector, we have two methods 
By using `.reshape()` and another by using `.newaxis` when slicing.

To convert into row vector
`x.reshape((1,3))`
`x[np.newaxis,:]`

Similarly to convert into column vector
`x.reshape((3,1))`
`x[:,np.newaxis]`

#### Array concatenation and splitting 
In this we'll learn how to concatenate multiple arrays into one array and splitting single array into multiple arrays.

##### Concatenation 
Concatenation or joining of two or more arrays can be accomplished through the routines 
`np.concatenate()`
`np.vstack()`
`np.hstack()`

Examples:-
`x=np.array([1,2,3])`
`y=np.array([4,5,6])`
`z=np.concatenate([x,y])`

We can also concatenate multiple arrays like
`arr=np.concatenate([x,y,x])`

To Concatenation of two dimensional arrays
`x=np.array([[1, 2, 3],[4, 5, 6]])

To concatenate along axis 0
`y=np.concatenate([x,x])`

To concatenate along axis 1
`y=np.concatenate([x,x],axis=1)`

We can use vstack and hstack for working with mixed dimension arrays
`np.vstack([x,y])`
`np.hstack([x,y])`

Similarly, np.dstack to concatenate along third axis.

##### Splitting of arrays 
This is opposite to concatenation and routines used are 
`np.split()`
`np.vsplit()`
`np.hsplit()`

For each of these, we pass a list of indices giving the splitting points 

`arr=np.arange(1,20)`
`x,y,z=np.split(arr,[5,9])`

Notice that N splits give N+1 sub arrays

Similarly vsplit and hsplit
`arr1=arr.reshape(4,5)`
`left,right=np.hsplit(arr1,[4])`

This splits the array vertically into two parts left and right

`arr1=arr.reshape(4,5)`
`upper,lower=np.vsplit(arr1,[3])`

This splits the array into two parts , upper and lower parts.







## Computations on Numpy arrays
The operations on Numpy arrays can be very fast or slow. The key to make it fast is to use vectorised operations using Numpy's Ufuncs (Universal Functions ) 

### Introducing Ufuncs
The vectorised operations can be implemented by simply performing an operation on an array, which is then applied to each element leading to faster implementation.

Divison of array by a scalar
`print(x/2)`
Each element is divided by 2

Similarly, we can divide array by an array
`print(x/y)`

We can also perform Ufuncs on multi dimensional arrays.

The vectorised implementation is much faster than a for loop and anytime you see a for loop

## Exploring Numpy's Ufuncs 
Ufuncs come in two flavours 
1. Unary which operate on a single input
2. Binary which operate on two inputs 

#### Array Arithmetic 
We can use standard +,-, multiplication and division,like
`x=np.arange(5)`
`print(x+5)`
`print(x/5)`
`print(x-4)`
`print(x*5)`

Negation
`print(-x)`
Modulus and exponential 
`print(x%3)`

We can write expressions using these operators
`print(-x+(4*x)+x/6))`

These are just wrappers to the built in numpy functions.

#### Absolute value
The pythons built in abs func
`abs(x)`

The Ufunc for this is `np.absolute()` or with the alias `np.abs()`

For complex number arrays, abs will give the magnitude.
`x = np.array([3 - 4j, 4 - 3j, 2 + 0j, 0 + 1j])`
`np.abs(x)`

#### Trigonometric Functions 

First we define an array of angles 
Three angles evenly spaced between 0 and π
`theta=np.linspace(0,np.pi,3)`

Now sin, cos, and tan
`print(sin(theta))`
`print(cos(theta))`
`print(tan(theta))`

Inverse trig funcs
`x=[1,0,7]`
`print(arcsin(x))`
`print(arccos(x))`
`print(arctan(x))`

#### Exponents and logarithms
Ufuncs available are 
`x=[1,2,3]`
`print(np.exp(x))`
`print('2^x',np.exp2(x))`
`print('3^x',np.power(3,x))`

#### Specialised Ufuncs 
These are available in script.special module.

## Advanced Ufuncs

##### Specifying output
We can specify where the output of the operation needs to be stored using the out parameter
`x=np.arange(5)`
`y=np.empty(5)`
`np.multiply(x,3,out=y)`


## Aggregations Min Max and everything in between

#### Summing values in an array
`x=[1,2,3,4]`
In python,
`sum(x)`

In numpy,
`np.sum(x)`

`np.sum()` works faster than `sum()` in python because it executes operation on compiler code.

The `sum` and `np.sum` are not identical

#### Minimum and Maximum
The Python syntax for min and Max is
`x=[1,2,5,8]`
`min(x)`
`max(x)`

In numpy, 
`np.min(x)`
`np.max(x)`

Difference between these two is that numpy functions operate faster.

A shorter syntax is to use methods of array objects itself, like
`x.min()`
`x.max()`

#### Multidimensional aggregates
One common aggregation operation is aggregate along a row or column.

Say you have a multi dimensional array,
`x=np.random.random((3,6))`

To compute the sum
`x.sum()`
By default, numpy Aggregations will return aggregate of the entire array.
There is a parameter to specify the aggregate along an *axis*

To get min value of each column along axis =0
`x.min(axis=0)`

Similarly to get max along each row
`x.max(axis=1)`

The *axis* keyword specifies the dimensions of the array that is collapsed rather than dimensions of the array.
axis=0 means the first axis will be collapsed for two dimensional array , i.e. columns are aggregated.

![[Screenshot_2023-10-16-10-21-53-836-edit_com.foxit.mobile.pdf.lite.jpg]]

## Computations on arrays: Broadcasting
Numpy's *Ufuncs* can be used to vectorise operations and remove slow python loops.
Similarly *broadcasting* is another means for vectorising operations.

Broadcasting is simply a set of rules to apply binary Ufuncs (addition, subtraction, multiplication etc) to arrays of different sizes.

#### Introducing broadcasting 

Recall that when we use operators like +,- etc on arrays of same sizes, these operations are performed element wise.
`x=np.arange(5)`
`y=np.random.randint(5)`
`print(x+y)`

Broadcasting allows us to add arrays of different sizes.
We can add a scalar(considered as a 0 D array) to the array and it gets added to each element of the array.
`print(a+6)`

We can think of this as an operation that stretches or duplicates 6 into array [6,6,6] and the added to get result.
The advantage of numpy Broadcasting is that it does not duplicate the array but it is a helpful mental model to understand broadcasting.

We can extend this to higher dimensional arrays.
`M=np.arange(24).reshape(4,6)`
`M+a`

The array a is broadcasted along the second dimension to match the shapr of the array M.

A more complicated example
`x=np.arange(6)`
`y=np.arange(7).reshape(7,1)`
Or
`y=np.arange(7)[:,np.newaxis]`

`x+y`
Here we've stretched both a and b to match a common shape and the result is a 2darray

![[Screenshot_2023-10-16-11-44-30-761-edit_com.foxit.mobile.pdf.lite.jpg]]
The light boxes represent the broadcasted values and no extra memory is allocated for these values.
These are useful to understand conceptually.

#### Broadcasting rules
Broadcasting follows a set of rules 

- Rule 1: If the two arrays differ in their number of dimensions, the shape of the
one with fewer dimensions is padded with ones on its leading (left) side.
- Rule 2: If the shape of the two arrays does not match in any dimension, the array
with shape equal to 1 in that dimension is stretched to match the other shape.
- Rule 3: If in any dimension the sizes disagree and neither is equal to 1, an error is
raised.

##### Broadcasting examples

Example 1
`M=np.ones((2,3))`
`a=np.arange(3)`
`M+a`

Example 2(we get an error in this example)
`M=np.ones((3,2))`
`a=np.arange(3)`
`M+a`

Example 3
`M=np.ones((3,1))`
`a=np.arange(3)`
`M+a`

This broadcast works for any binary Ufuncs and not just addition.

#### Broadcasting in practise
##### Centering an array
We have observations as a 10x3 matrix

We computed the mean of this matrix along first dimension (0)
`X_mean=X.mean(0)`

Now subtract observations from mean
`X_centered=X-X_mean`


### Comparisons, Masks and boolean logic
Masking is used to extract, modify, count or manipulate based on certain criterion.

#### Comparison operators as Ufuncs
Earlier we've discussed about arithmetic operations as Numpy Ufuncs. Numpy also has comparison operators as element wise Ufuncs.

They are:-
1. <
2. >
3. <=
4. >=
5. ==
6. !=
The result is these operations is a Boolean array.

We can also perform element wise comparision of 2 arrays like
`(2*x)==(x**2)`

When the less than operator is used, numpy uses the Ufuncs `np.less()` internally.
![[Screenshot_2023-10-16-19-32-24-639-edit_com.foxit.mobile.pdf.lite.jpg]]
These can also be used for multi dimensional arrays.

#### Working with Boolean arrays
`x=np.arange(15).reshape((3,5))`

Now to count the no of True entries >6, we use 
`np.count_nonzero(x>6)`
Or
`np.sum(x>6)`
Here False is interpreted as 0 and True as 1.
This 🔝 method is useful to get the entries along any row or column as

`np.sum(x>6,axis=1)`

We can also use `np.any()` and `np.all()` to check whether we have True entries satisfying a certain condition.

`np.any(x>7)`
`np.all(x<15)`

We can use this along a particular axis
`np.any(x<6,axis=1)`

Make sure to use `np.sum()`, `np.any()` and `np.all()` when working with multi dimensional arrays bcz pythons built in functions such as `sum,any,all` will produce unintended results.

##### Boolean operators
We can use Python's bitwise operators &,|,~,^ for numpy arrays where these are implemented element wise using Ufuncs internally.





![[Screenshot_2023-10-17-10-02-46-471-edit_com.foxit.mobile.pdf.lite.jpg]]

#### Boolean arrays as masks
We know how to get a Boolean array based on certain condition.
`x=np.arange(8).reshape(4,2)`
`x<5`
Will give a Boolean array and to get the values or elements of the array which satisfy this condition, we use masking
`x[x<5]`

To get these elements, we can simply index this Boolean array on this condition and this is known as *masking*.
Here we get all those values which have True values in this boolean array.

#### Difference of when to use *and or* keywords and *&*|

These only difference between *and,or* and *&* | is that *and,or* gauge the truth or falsehood of entire object whereas *&* | refers to bits within each object.

When you use *and* or *or*, it's equivalent to asking python to treat object as a single Boolean entity 
When *&* or | is used on integers, it operates on bits of the element, using the *and* or *or* to individual bits making the number.

Remember to always use & | for arrays to compare elements element wise using and of will give an error.

## Fancy indexing
Fancy indexing is just like simple indexing but we pass an array of indices instead of a single scalar.

### Exploring fancy indexing
Fancy indexing is simple , it means to pass array of indices to acces multiple elements at once.

We create an array as 
`import numpy as np`
`rand=np.random.RandomState(42)`
`x=rand.randint(100,size=10)`

Fancy indexing as follows:-
	Method 1
	`[x[5],x[4],x[6]]`
	We get elements of the array.
	Method 2
	We can also pass an array of indicies to obtain same results like
	`ind=[1,4,7]`
	`x[Ind]`

With fancy indexing, the shape of the result reflects the shape the index arrays rather than the shape of the array being indexed.
`ind=[[3,4],`
	   `[1,3]]`
`x[ind]`

Will give us an array of the given index.

We can also use fancy indexing for arrays of Multiple dimensions like
`row=[3,0,1]`
`col=[2,1,0]`

`x[row,col]`
First is for row and next is for column.

Here broadcasting works.

#### Combining indexing
We can combine fancy indexing with simple indexing, slicing and masking.

Simple indexing
`x=np.arange(12).reshape(3,4)`
`x[2,[3,0,1]]`
This means to get the 3rd , 0th and 1st column values of 2nd row.

Slicing 
`x[1:,[3,0,2]]`

Examples of fancy indexing to be learnt later.

## Sorting arrays
*Selection sort* finds the minimum of the list and swaps till the list is sorted.

### Fast sorting in numpy np.sort() np.argsort()
Pythons built in routines such as `.sort()` and `.sorted()` to work with lists but we don't use this for numpy arrays.

Numpy's `np.sort()` is efficient for dealing with numpy arrays.
`np.sort()` uses the O(NlogN) time complexity 
`.heapsort()`
`.quicksort()`
`mergesort()`

But `quicksort()` is used mostly.

`x=np.array([2,5,3,2,1])`
`np.sort(x)`

`i=np.argsort(x)`
Returns the indicies of the sorted elements.

`x[i]`
Returns the sorted array via fancy indexing.

#### Sorting arrays along row or column

`rand=np.random.RandomState(42)`
`x=np.rand.randint(15,size=(3,5))`

To sort along each column,
`np.sort(x,axis=0)`

To sort along each row,
`np.sort(x,axis=1)`

### Partial sorts Partitioning 
Sometimes we want to find the K smallest elements of the array. Numpy provides `np.partition()` 

`np.partition()` takes an array and a number K, and returns a new array with the K smallest elements to the left and other elements to the right in arbitrary order.
`x=np.random.randint(10,size=(2,5))`
`np.partition(x,3)`

The first three values in the array are the 3 smallest in the array.

`np.partition(X,2,axis=1)`
The result of this is the first two slots in each row contains the smallest values from that row
