/*!
 * \file kd_tree.cc
 * \brief Implements KDTree class.
 * \author Boyoon Jung (boyoon@robotics.usc.edu)
 */
#include "kd_tree.h"
using namespace bj;

#include <gsl/gsl_blas.h>

#include <cmath>
#include <values.h>

// header files for debugging
#include <iostream>
using std::cerr;
using std::endl;


// utility functions
inline bool gsl_vector_equal(gsl_vector* x, gsl_vector* y)
{
    if (x->size != y->size)
	return false;

    for (int i=0; i<x->size; i++)
	if (gsl_vector_get(x,i) != gsl_vector_get(y,i))
	    return false;

    return true;
}


// constructor
KDTree::KDTree(gsl_vector* bsize) : tree_size(0), _root(0)
{
    // store parameters
    bin_size = gsl_vector_alloc(bsize->size);
    gsl_vector_memcpy(bin_size, bsize);

    // initialize variables
    bin_index = gsl_vector_alloc(bin_size->size);
}


// destructor
KDTree::~KDTree(void)
{
    // de-allocate objects
    gsl_vector_free(bin_size);
    gsl_vector_free(bin_index);

    // de-allocate a kd-tree
    destroy_tree(_root);
}


// add a point to the current kd-tree
void KDTree::append_node(kd_node*& node, gsl_vector* point)
{
    // root node ?
    if (node == 0)
    {
	node = kd_node_alloc(point);
	node->split = 0;
	return;
    }

    // matched ?
    if (gsl_vector_equal(node->element, point))
    {
	node->num_points++;
	return;
    }

    // belong to the left tree ?
    if (gsl_vector_get(node->element,node->split) <= gsl_vector_get(point,node->split))
    {
	if (node->left)
	    append_node(node->left, point);
	else
	{
	    node->left = kd_node_alloc(point);
	    node->left->split = find_split(node->element, point, node->split);
	}
    }

    // belong to the right tree
    else
    {
	if (node->right)
	    append_node(node->right, point);
	else
	{
	    node->right = kd_node_alloc(point);
	    node->right->split = find_split(node->element, point, node->split);
	}
    }
}


// destroy a kd-tree
void KDTree::destroy_tree(kd_node*& tree)
{
    // empty tree ?
    if (! tree) return;

    // destroy child kd-trees
    if (tree->left) destroy_tree(tree->left);
    if (tree->right) destroy_tree(tree->right);

    // de-allocate the top node
    kd_node_free(tree);
}


// insert a point into a tree
void KDTree::insert(gsl_vector* x)
{
    // discretize the point (into a bin index)
    for (int i=0; i<bin_index->size; i++)
	gsl_vector_set(bin_index,i,
		floor(gsl_vector_get(x,i)/gsl_vector_get(bin_size,i)));

    // add the index to the current kd-tree
    append_node(_root, bin_index);
}




// a constructor
MRKDTree::MRKDTree(gsl_vector** P, int size, double mbw)
	: _root(0), tree_size(0), mb_width(0), dimension(0),
	  centroid(0), cov(0), hr_min(0), hr_max(0), tmpv(0)
{
    // build a tree if data points are given
    if (P)
	this->build(P, size, mbw);
}


// a destructor
MRKDTree::~MRKDTree(void)
{
    // de-allocate a mrkd-tree
    destroy_tree(_root);
}


// build a mrkd-tree
void MRKDTree::build_tree(mrkd_node*& node, vector<unsigned int>& index, gsl_vector** P)
{
    // create a new node
    node = mrkd_node_alloc();
    node->num_points = index.size();
    
    // if the node is a singleton
    if (index.size() == 1)
    {
	gsl_vector_memcpy(node->centroid, P[index[0]]);
	gsl_matrix_set_zero(node->cov);
	gsl_vector_memcpy(node->hr_min, P[index[0]]);
	gsl_vector_memcpy(node->hr_max, P[index[0]]);
    }
    else
    {
	// create the bounding box
	compute_bbox(index, P);
	gsl_vector_memcpy(node->hr_min, hr_min);
	gsl_vector_memcpy(node->hr_max, hr_max);

	// check if it is a leaf node
	gsl_vector_sub(hr_max, hr_min);
	if (gsl_vector_max(hr_max) <= mb_width)
	{
	    // compute the statistics
	    compute_stats(index, P);
	    gsl_vector_memcpy(node->centroid, centroid);
	    gsl_matrix_memcpy(node->cov, cov);
	}

	// otherwise, split the hyper cube
	else
	{
	    // compute the splitting hyper plane
	    node->split_dim = gsl_vector_max_index(hr_max);
	    node->split_val = gsl_vector_get(node->hr_min, node->split_dim) +
			      (gsl_vector_get(node->hr_max, node->split_dim) -
			       gsl_vector_get(node->hr_min, node->split_dim)) / 2;

	    // split data points into two sets
	    vector<unsigned int> left_index;
	    vector<unsigned int> right_index;
	    left_index.reserve(index.size());
	    right_index.reserve(index.size());

	    for (vector<unsigned int>::iterator i=index.begin(); i!=index.end(); i++)
	    {
		if (gsl_vector_get(P[*i], node->split_dim) < node->split_val)
		    left_index.push_back(*i);
		else
		    right_index.push_back(*i);
	    }

	    // make child nodes
	    build_tree(node->left, left_index, P);
	    build_tree(node->right, right_index, P);
	}
    }
}


// destroy a mrkd-tree
void MRKDTree::destroy_tree(mrkd_node*& tree)
{
    // empty tree ?
    if (! tree) return;

    // destroy child kd-trees
    if (tree->left) destroy_tree(tree->left);
    if (tree->right) destroy_tree(tree->right);

    // de-allocate the top node
    mrkd_node_free(tree);
}


// compute the bound box
void MRKDTree::compute_bbox(vector<unsigned int>& index, gsl_vector** P)
{
    gsl_vector_set_all(hr_min, MAXDOUBLE);
    gsl_vector_set_all(hr_max, -MAXDOUBLE);

    for (vector<unsigned int>::iterator i=index.begin(); i!=index.end(); i++)
    {
	for (int d=0; d<dimension; d++)
	{
	    if (gsl_vector_get(P[*i], d) < gsl_vector_get(hr_min, d))
		gsl_vector_set(hr_min, d, gsl_vector_get(P[*i], d));

	    if (gsl_vector_get(P[*i], d) > gsl_vector_get(hr_max, d))
		gsl_vector_set(hr_max, d, gsl_vector_get(P[*i], d));
	}
    }
}


// compute the statistics
void MRKDTree::compute_stats(vector<unsigned int>& index, gsl_vector** P)
{
    // compute the mean vector
    gsl_vector_set_zero(centroid);
    for (vector<unsigned int>::iterator i=index.begin(); i!=index.end(); i++)
	gsl_vector_add(centroid, P[*i]);
    gsl_vector_scale(centroid, 1.0/index.size());

    // compute the covariance matrix
    gsl_matrix_set_zero(cov);
    for (vector<unsigned int>::iterator i=index.begin(); i!=index.end(); i++)
    {
	gsl_vector_memcpy(tmpv, P[*i]);
	gsl_vector_sub(tmpv, centroid);
	gsl_blas_dsyr(CblasUpper, 1.0, tmpv, cov);

	/*
	for (int m=1; m<dimension; m++)
	    for (int n=0; n<m; n++)
		gsl_matrix_set(cov, m,n, gsl_matrix_get(cov, n,m));
	*/
    }
    gsl_matrix_scale(cov, 1.0/index.size());
}


// build a multiresolution KD-tree
void MRKDTree::build(gsl_vector** P, int size, double mbw)
{
    // remove the old tree if exists
    if (_root)
	destroy_tree(_root);

    // determine the dimesion of a data point
    if (size < 1)
	return;
    else
	dimension = P[0]->size;

    // allocate temporary buffers
    if (centroid) gsl_vector_free(centroid);
    centroid = gsl_vector_alloc(dimension);

    if (cov) gsl_matrix_free(cov);
    cov = gsl_matrix_alloc(dimension, dimension);

    if (hr_min) gsl_vector_free(hr_min);
    hr_min = gsl_vector_alloc(dimension);

    if (hr_max) gsl_vector_free(hr_max);
    hr_max = gsl_vector_alloc(dimension);

    if (tmpv) gsl_vector_free(tmpv);
    tmpv = gsl_vector_alloc(dimension);

    // determine the minimum bounding-box width
    if (mbw >= 0)
	mb_width = mbw;
    else
    {
	// compute the bounding box
	gsl_vector_set_all(hr_min, MAXDOUBLE);
	gsl_vector_set_all(hr_max, -MAXDOUBLE);
	for (int i=0; i<size; i++)
	    for (int d=0; d<dimension; d++)
	    {
		if (gsl_vector_get(P[i], d) < gsl_vector_get(hr_min, d))
		    gsl_vector_set(hr_min, d, gsl_vector_get(P[i], d));

		if (gsl_vector_get(P[i], d) > gsl_vector_get(hr_max, d))
		    gsl_vector_set(hr_max, d, gsl_vector_get(P[i], d));
	    }

	// compute the minimum bounding-box width
	gsl_vector_sub(hr_max, hr_min);
	mb_width = gsl_vector_min(hr_max) * -mbw / 100.0;
    }

    // create indices
    vector<unsigned int> index;
    index.reserve(size);
    for (int i=0; i<size; i++)
	index.push_back(i);

    // build a tree recursively
    this->build_tree(_root, index, P);
}
