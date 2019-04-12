/*!
 * \file kd_tree.h
 * \brief Provides kernel-density tree data structures.
 * \author Boyoon Jung (boyoon@robotics.usc.edu)
 */
#ifndef __KD_TREE_H
#define __KD_TREE_H

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

#include <vector>
using std::vector;


namespace bj
{


//! Data structure for a node of a KD-tree.
struct kd_node
{
    gsl_vector* element;	//!< A splitting point.
    int split;			//!< A splitting dimension.

    int num_points;		//! The number of points owned by this leaf node.

    kd_node* left;		//!< A left kd-tree.
    kd_node* right;		//!< A right kd-tree.
};


//! Data structure for a node of a multiresolution KD-tree.
struct mrkd_node
{
    int split_dim;		//!< A splitting dimension.
    double split_val;		//!< A splitting value.

    int num_points;		//!< The number of points owned by this sub-tree.
    gsl_vector* hr_min;		//!< The bounding hyper-rectangle of the points below this node.
    gsl_vector* hr_max;		//!< The bounding hyper-rectangle of the points below this node.

    gsl_vector* centroid;	//!< The centroid of the points owned by this leaf node.
    gsl_matrix* cov;		//!< The covariance of the points owned by this leaf node.

    mrkd_node* left;		//!< A left mrkd-tree.
    mrkd_node* right;		//!< A right mrkd-tree.
};


/*!
 * \addtogroup machine_learning
 * \{
 */

//! A KD-Tree data structure class.
/*!
 * TDTree class provies an efficient way to count the number of bins
 * where data points fall into. This class is written mainly for
 * AdaptiveParticleFilter class.
 */
class KDTree
{
    private:
	int tree_size;			// size of a kd-tree
	kd_node* _root;			// root node of a kd-tree

	gsl_vector* bin_size;		// size of bins
	gsl_vector* bin_index;		// bin index

	// add a point to the current kd-tree
	void append_node(kd_node*& node, gsl_vector* point);

	// destroy a kd-tree
	void destroy_tree(kd_node*& tree);

	// allocate a node
	inline kd_node* kd_node_alloc(gsl_vector* point);

	// de-allocate a node
	inline void kd_node_free(kd_node*& n);

	// find a proper split
	inline int find_split(gsl_vector* x, gsl_vector* y, int s);

    public:
	//! A contructor.
	KDTree(gsl_vector* bin_size);

	//! A destructor.
	~KDTree(void);

	//! Insert a datum into a tree.
	void insert(gsl_vector* x);

	//! Remove all nodes, and initialize a tree.
	void reset(void) { destroy_tree(_root); }

	//! Return the root node of a tree.
	kd_node* root(void) { return _root; }

	//! Return the current tree size.
	int size(void) { return tree_size; }

	//! Return the bin size.
	gsl_vector* binSize() { return bin_size; }
};

//! \}


// allocate a node
kd_node* KDTree::kd_node_alloc(gsl_vector* point)
{
    kd_node* n = new kd_node;
    n->element = gsl_vector_alloc(point->size);
    gsl_vector_memcpy(n->element, point);
    n->num_points = 1;
    n->left = n->right = 0;
    tree_size++;
    return n;
}


// de-allocate a node
void KDTree::kd_node_free(kd_node*& n)
{
    gsl_vector_free(n->element);
    delete n;
    n = 0;
    tree_size--;
}


// find a proper split
int KDTree::find_split(gsl_vector* x, gsl_vector* y, int s)
{
    // get the next dimension (simple!)
    if (++s < x->size)
	return s;
    else
	return 0;		// roll-over
}


/*!
 * \addtogroup machine_learning
 * \{
 */

//! A multiresolution KD-Tree data structure class.
/*!
 * MRTDTree class is written mainly for VFEMCluster class.
 * For efficiency, the statistics are computed only for leaf nodes.
 */
class MRKDTree
{
    private:
	mrkd_node* _root;		// root node of a mrkd-tree
	int tree_size;			// size of a kd-tree
	double mb_width;		// minimum bounding-box width
	int dimension;			// dimension of a data point

	gsl_vector* centroid;		// temporary buffers
	gsl_matrix* cov;
	gsl_vector* hr_min;
	gsl_vector* hr_max;
	gsl_vector* tmpv;

	// build a mrkd-tree recursively
	void build_tree(mrkd_node*& node, vector<unsigned int>& index, gsl_vector** P);

	// destroy a kd-tree
	void destroy_tree(mrkd_node*& tree);

	// allocate a node
	mrkd_node* mrkd_node_alloc(void);

	// de-allocate a node
	void mrkd_node_free(mrkd_node*& node);

	// compute the bounding box
	void compute_bbox(vector<unsigned int>& index, gsl_vector** P);

	// compute the statistics
	void compute_stats(vector<unsigned int>& index, gsl_vector** P);

    public:
	//! A contructor.
	/*!
	 * \param P A set of data points.
	 * \param size The number of data points in \a P.
	 * \param mbw The minimum bounding-box width.
	 * 	If negative, the unit of \a mbw is percentage.
	 */
	MRKDTree(gsl_vector** P=0, int size=0, double mbw=-1.0);

	//! A destructor.
	~MRKDTree(void);

	//! build a multiresolution KD-tree.
	/*!
	 * \param P A set of data points.
	 * \param size The number of data points in \a P.
	 * \param mbw The minimum bounding-box width.
	 * 	If negative, the unit of \a mbw is percentage.
	 */
	void build(gsl_vector** P, int size, double mbw=-1.0);

	//! Return the root node of a tree.
	mrkd_node* root(void) { return _root; }

	//! Return the current tree size.
	int size(void) { return tree_size; }
};

//! \}


// allocate a node
inline mrkd_node* MRKDTree::mrkd_node_alloc(void)
{
    mrkd_node* node = new mrkd_node;
    node->centroid = gsl_vector_alloc(dimension);
    node->cov = gsl_matrix_alloc(dimension, dimension);
    node->hr_min = gsl_vector_alloc(dimension);
    node->hr_max = gsl_vector_alloc(dimension);
    node->left = 0;
    node->right = 0;
    tree_size++;
    return node;
}


// de-allocate a node
inline void MRKDTree::mrkd_node_free(mrkd_node*& node)
{
    gsl_vector_free(node->centroid);
    gsl_matrix_free(node->cov);
    gsl_vector_free(node->hr_min);
    gsl_vector_free(node->hr_max);
    delete node;
    node = 0;
    tree_size--;
}


}	// namespace

#endif	// __KD_TREE_H
