from . cimport _arma as arma
from . cimport _arma_numpy as arma_numpy
#from cli cimport CLI
#from cli cimport SetParam, SetParamPtr, SetParamWithInfo, GetParamPtr
#from cli cimport EnableVerbose, DisableVerbose, DisableBacktrace, ResetTimers, EnableTimers
from sklearn.utils import check_array
from ._serialization cimport SerializeIn, SerializeOut

import numpy as np
cimport numpy as np

from libcpp.string cimport string
from libcpp cimport bool
from libcpp.vector cimport vector
from libc.stdint cimport SIZE_MAX

from cython.operator import dereference
cimport cython

cdef extern from "<mlpack/methods/det/dtree.hpp>" namespace "mlpack::det" nogil:
    cdef cppclass DTree[MatType]:
        DTree() nogil except +
        DTree(const DTree& obj) nogil except +
        DTree(MatType& data) nogil except +
        DTree(
            const arma.Col[double]& maxVals,
            const arma.Col[double]& minVals,
            const size_t totalPoints
        ) nogil except +
        double Grow(
            MatType& data,
            arma.Col[size_t]& oldFromNew,
            const bint useVolReg,
            const size_t maxLeafSize,
            const size_t minLeafSize
        ) nogil
        double PruneAndUpdate(
            const double oldAlpha,
            const size_t points,
        ) nogil
        # Getters (modified by removing implementations)
        DTree* const Left() nogil
        DTree* const Right() nogil
        arma.Col[double]& MaxVals() nogil
        arma.Col[double]& MinVals() nogil
        size_t Start() nogil
        size_t End() nogil
        bint Root() nogil
        size_t NumChildren() nogil
        size_t SplitDim() nogil
        double SplitValue() nogil

cdef size_t cNumLeaves(DTree[arma.Mat[double]]* tree):
    if tree.NumChildren() == 0:
        return 1
    else:
        return cNumLeaves(tree.Left()) + cNumLeaves(tree.Right())

cdef size_t cNumNodes(DTree[arma.Mat[double]]* tree):
    if tree.NumChildren() == 0:
        return 1
    else:
        # Include internal node in count
        return 1 + cNumNodes(tree.Left()) + cNumNodes(tree.Right())

cdef size_t cDepth(DTree[arma.Mat[double]]* tree):
    if tree.NumChildren() == 0:
        return 1
    else:
        left_depth = cDepth(tree.Left())
        right_depth = cDepth(tree.Right())
        if left_depth > right_depth:
            return left_depth + 1
        else:
            return right_depth + 1

cdef double cPrune(DTree[arma.Mat[double]]* tree, double alpha, size_t total_points, size_t max_leaf_nodes, size_t max_depth):
    # Prune until fits criteria
    while cNumLeaves(tree) > max_leaf_nodes or cDepth(tree) > max_depth:
        alpha = tree.PruneAndUpdate(alpha, total_points)
    return alpha

cdef size_t cGetArrayedTree(DTree[arma.Mat[double]]* tree,
                            np.ndarray[np.intp_t, ndim=1] meta,
                            np.ndarray[np.intp_t, ndim=1] feature,
                            np.ndarray[np.double_t, ndim=1] threshold,
                            np.ndarray[np.intp_t, ndim=1] children_left,
                            np.ndarray[np.intp_t, ndim=1] children_right):
    # Create node index
    cdef size_t cur_i = meta[0] # Extract from array (a simple way of passing around values)
    meta[0] += 1
 
    if tree.NumChildren() > 0:
        feature[cur_i] = tree.SplitDim()
        threshold[cur_i] = tree.SplitValue()
        children_left[cur_i] = cGetArrayedTree(
            tree.Left(), meta, feature, threshold, children_left, children_right)
        children_right[cur_i] = cGetArrayedTree(
            tree.Right(), meta, feature, threshold, children_left, children_right)

    # Else return this node's index value
    return cur_i


cdef class PyDTree:
    cdef DTree[arma.Mat[double]]* modelptr

    def __cinit__(self, X=None, new=True, max_vals=None, min_vals=None, total_points=None):
        if new == False:
            return # Used for left() and right() below
        elif min_vals is not None and max_vals is not None and total_points is not None:
            max_vals_arma = dereference(arma_numpy.numpy_to_col_d(max_vals, False))  # Make copy
            min_vals_arma = dereference(arma_numpy.numpy_to_col_d(min_vals, False))  # Make copy
            self.modelptr = new DTree[arma.Mat[double]](max_vals_arma, min_vals_arma, total_points)
            # print(self.min_vals())
            # print(self.max_vals())
        elif X is not None:
            X_arma = dereference(
                arma_numpy.numpy_to_mat_d(X, False))  # Make copy
            self.modelptr = new DTree[arma.Mat[double]](X_arma)
        else:
            self.modelptr = new DTree[arma.Mat[double]]()

    def __dealloc__(self):
        del self.modelptr

    def __getstate__(self):
        return SerializeOut(self.modelptr, "DTree[]")

    def __setstate__(self, state):
        SerializeIn(self.modelptr, state, "DTree[]")

    def __reduce_ex__(self, version):
        return (self.__class__, (), self.__getstate__())

    def __str__(self):
        return self.get_tree_str()

    def max_vals(self):
        # Have to remove "const" qualifier but must be lvalue (i.e. persistent) so use temp variable
        temp = (<arma.Col[double]>self.modelptr.MaxVals()) 
        return arma_numpy.col_to_numpy_d(temp)

    def min_vals(self):
        # Have to remove "const" qualifier but must be lvalue (i.e. persistent) so use temp variable
        temp = (<arma.Col[double]>self.modelptr.MinVals()) 
        return arma_numpy.col_to_numpy_d(temp)

    def fit(self, np.ndarray X,
            size_t max_depth=SIZE_MAX,
            size_t max_leaf_nodes=SIZE_MAX,
            size_t min_leaf_size=1,
            ):
        X = check_array(X, dtype=np.float)
        cdef size_t total_points = X.shape[0]
        X_arma = dereference(
            arma_numpy.numpy_to_mat_d(X, True))
        old_new_map_arma = dereference(
            arma_numpy.numpy_to_col_s(np.arange(X.shape[0], dtype=np.int), True))  # No need to make copy
        
        # Fully grow tree
        # Set max_leaf_size to ensure that the tree grows fully based on min_leaf_size
        # if(leaf_size > max_leaf_size), then the node is split
        # Thus, min_leaf_size*2-1 ensures this condition (making the interface behave similarly to sklearn min_leaf_size)
        cdef size_t max_leaf_size = min_leaf_size * 2 - 1
        cdef double alpha = self.modelptr.Grow(X_arma, old_new_map_arma, False, max_leaf_size, min_leaf_size)
        #import warnings
        #warnings.warn('Before max_leaf_size=%d, min_leaf_size=%d, max_depth=%d, max_leaf_nodes=%d, n_leaves=%d, depth=%d'
        #              % (max_leaf_size, min_leaf_size, max_depth, max_leaf_nodes, cNumLeaves(self.modelptr), cDepth(self.modelptr)))

        # Then prune tree based on max_leaf_nodes or max_depth
        alpha = cPrune(self.modelptr, alpha, total_points, max_leaf_nodes, max_depth)

        #warnings.warn('After max_leaf_size=%d, min_leaf_size=%d, max_depth=%d, max_leaf_nodes=%d, n_leaves=%d, depth=%d'
        #              % (max_leaf_size, min_leaf_size, max_depth, max_leaf_nodes, cNumLeaves(self.modelptr), cDepth(self.modelptr)))
        return self

    def grow(self, np.ndarray X, size_t max_leaf_size=10, size_t min_leaf_size=5):
        X_arma = dereference(
            arma_numpy.numpy_to_mat_d(X, False))  # Make copy
        old_new_map_arma = dereference(
            arma_numpy.numpy_to_col_s(np.arange(X.shape[0], dtype=np.int), True))  # No need to make copy
        alpha = self.modelptr.Grow(X_arma, old_new_map_arma, False, max_leaf_size, min_leaf_size)
        return alpha

    def prune_and_update(self, double old_alpha, size_t points):
        return self.modelptr.PruneAndUpdate(old_alpha, points)
    
    def left(self):
        subtree = PyDTree(new=False)
        subtree.modelptr = new DTree[arma.Mat[double]](dereference(self.modelptr.Left()))
        return subtree

    def right(self):
        subtree = PyDTree(new=False)
        subtree.modelptr = new DTree[arma.Mat[double]](dereference(self.modelptr.Right()))
        return subtree

    def get_tree_str(self, level=0, is_left=None, show_leaves=True):
        # Determine if leaf
        is_leaf = self.num_children() == 0
        if is_leaf and not show_leaves:
            return ''

        node_str = 'Root' if is_left is None else 'Left' if is_left else 'Right'
        node_str += '(%d)' % (self.modelptr.End() - self.modelptr.Start())
        node_str += ' min=%s' % str(self.min_vals())
        node_str += ' max=%s' % str(self.max_vals())

        if is_leaf:
            node_str += ': Leaf node'
            left_str = ''
            right_str = ''
        else:
            node_str += (': split_dim=%d, split_value=%g' 
                        % (self.split_dim(), self.split_value()))
            left_str = self.left().get_tree_str(level=level+1, is_left=True, show_leaves=show_leaves)
            right_str = self.right().get_tree_str(level=level+1, is_left=False, show_leaves=show_leaves)

        out_str = '  ' * level + node_str + '\n' + left_str + right_str
        return out_str

    def get_arrayed_tree(self):
        """Get an arrayed tree like the scikit learn tree.
        In particular, we need the following parallel arrays:
        `feature`
        `threshold`
        `children_left`
        `children_right`
        """
        # Init variables
        cdef size_t max_num_nodes = cNumNodes(self.modelptr)
        cdef size_t max_index = 0
        cdef np.ndarray[np.intp_t, ndim=1] meta = np.zeros(1, dtype=np.intp)
        cdef np.ndarray[np.intp_t, ndim=1] feature = -np.ones(max_num_nodes, dtype=np.intp)
        cdef np.ndarray[np.float_t, ndim=1] threshold = np.nan*np.ones(max_num_nodes, dtype=np.float)
        cdef np.ndarray[np.intp_t, ndim=1] children_left = -np.ones(max_num_nodes, dtype=np.intp)
        cdef np.ndarray[np.intp_t, ndim=1] children_right = -np.ones(max_num_nodes, dtype=np.intp)

        # Get arrays
        cGetArrayedTree(self.modelptr, meta, feature, threshold, children_left, children_right)

        # Output arrayed tree
        n_features = len(self.min_vals())
        return _ArrayedTree(meta[0], n_features, feature, threshold, children_left, children_right)


    def depth(self):
        return cDepth(self.modelptr)
    
    def num_leaves(self):
        return cNumLeaves(self.modelptr)

    cpdef bool root(self):
        return self.modelptr.Root()

    cpdef size_t num_children(self):
        return self.modelptr.NumChildren()

    cpdef size_t split_dim(self):
        return self.modelptr.SplitDim()

    cpdef double split_value(self):
        return self.modelptr.SplitValue()


class _ArrayedTree():
    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    def __init__(self,
                 np.intp_t node_count,
                 np.intp_t n_features,
                 np.ndarray[np.intp_t, ndim=1] feature,
                 np.ndarray[np.float_t, ndim=1] threshold,
                 np.ndarray[np.intp_t, ndim=1] children_left,
                 np.ndarray[np.intp_t, ndim=1] children_right):
        self.n_features = n_features
        self.feature = feature[:node_count]
        self.threshold = threshold[:node_count]
        self.children_left = children_left[:node_count]
        self.children_right = children_right[:node_count]
