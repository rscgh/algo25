from scipy.sparse import csr_matrix
import numpy as np
from nilearn.surface import load_surf_mesh
from scipy.sparse import issparse
from nilearn import datasets, surface
from scipy import sparse

fsaverage = datasets.fetch_surf_fsaverage('fsaverage')


def _compute_adjacency_matrix(surface, values="ones", dtype=None):
    from scipy.sparse import csr_matrix

    surface = load_surf_mesh(surface)
    # This is a bit of a hack to quickly find a unique set of all edges.
    n = surface.coordinates.shape[0]
    edges = np.vstack(
        [
            surface.faces[:, [0, 1]],
            surface.faces[:, [0, 2]],
            surface.faces[:, [1, 2]],
        ]
    )
    edges = edges.astype(np.int64)
    bigcol = edges[:, 0] > edges[:, 1]
    lilcol = ~bigcol
    edges = np.concatenate(
        [
            edges[bigcol, 0] + edges[bigcol, 1] * n,
            edges[lilcol, 1] + edges[lilcol, 0] * n,
        ]
    )
    edges = np.unique(edges)
    (u, v) = (edges // n, edges % n)
    edge_lens = np.ones_like(edges)
    
    # We can now make a sparse matrix.
    ee = np.concatenate([edge_lens, edge_lens])
    uv = np.concatenate([u, v])
    vu = np.concatenate([v, u])
    return csr_matrix((ee, (uv, vu)), shape=(n, n))




def spherical_dists(rows, cols, spherical_surf, sphere_max=None):

    if sphere_max is None:
        radius = spherical_surf.coordinates.max(); 100 
    
    # first normalize to unit sphere
    coords1=spherical_surf.coordinates[rows]/radius;
    coords2=spherical_surf.coordinates[cols]/radius;
    
    # calculate the distances using efficient vector operations
    dot_product = np.einsum('ij,ij->i', coords1, coords2)
    spherical_distances = np.arccos(np.clip(dot_product, -1.0, 1.0)) * radius
    return spherical_distances
    

def get_multi_hop_neighbours(adjacency_matrix, spherical_surf, n_hop_iter=10, max_neigh_dist=7, v=False):
    multi_hop_adj = adjacency_matrix
    
    for hn in range(n_hop_iter):
        if v: print("### hop nr:", hn) 

        # get all the next hop neightbours based on the current adjacency relations
        multi_hop_adj = adjacency_matrix.dot(multi_hop_adj)
        # get all neighbour pair indices, len(rows)==len(cols)==n_pairs
        rows, cols = multi_hop_adj.nonzero()
        if v: print(len(rows), len(cols), issparse(multi_hop_adj))

        # get the spherical distances between all the corrdinates for each vertex pair
        spherical_distances = spherical_dists(rows,cols, spherical_surf);
        if v: print("distances:", spherical_distances.shape, spherical_distances.max(), spherical_distances.mean())
        
        # only keep the neighbours within a given distance
        keep= spherical_distances <=max_neigh_dist        
        filtered_rows = rows[keep]
        filtered_cols = cols[keep]
        filtered_data = np.ones(len(filtered_cols))
        # create a new sparse matrix with the filtered neighbour pairs
        multi_hop_adj = csr_matrix((filtered_data, (filtered_rows, filtered_cols)), shape=multi_hop_adj.shape)
        if v: print("filtered:", len(multi_hop_adj.nonzero()[0]))

    # enforce unqiueness of pairs
    rows, cols = multi_hop_adj.nonzero()
    unique_pairs = np.unique(np.vstack([rows, cols]), axis=1)
    filtered_data = np.ones(unique_pairs.shape[1])
    uniq_multi_hop_adj = csr_matrix((filtered_data, (unique_pairs[0], unique_pairs[1])), \
                                    shape=multi_hop_adj.shape)
    spherical_distances = spherical_dists(unique_pairs[0],unique_pairs[1], spherical_surf);
    
    return uniq_multi_hop_adj, spherical_distances


def surf_smooth(data ,fwhm=4, surf=fsaverage.sphere_left, \
                multi_hop_adj=None, dists = None, weight_mat=None, **kwargs):
    
    surf = load_surf_mesh(surf)
    if multi_hop_adj is None:
        
        adj = _compute_adjacency_matrix(surf, values="ones")
        multi_hop_adj, dists = get_multi_hop_neighbours(adj, surf, **kwargs)

    rows, cols = multi_hop_adj.nonzero()
    
    if dists is None:    
        dists = spherical_dists(rows,cols, surf);

    if weight_mat is None:
        #fwhm=2*1.8
        sigma = fwhm / 2.3548  # Convert FWHM to sigma (standard deviation)
        weights = np.exp(-0.5 * (dists ** 2) / (sigma ** 2))  # Gaussian weight
        weight_mat = csr_matrix((weights, (rows, cols)), shape=multi_hop_adj.shape)

        row_sums = np.array(weight_mat.sum(axis=1)).flatten()  # Convert to 1D array
        # Replace zeros in row sums to avoid division by zero
        row_sums[row_sums == 0] = 1
        # Compute inverse of row sums
        inv_row_sums = 1.0 / row_sums
        # Create a diagonal matrix of the inverse row sums
        D_inv = csr_matrix((inv_row_sums, (np.arange(len(inv_row_sums)), np.arange(len(inv_row_sums)))), shape=(weight_mat.shape[0], weight_mat.shape[0]))
        # Normalize the matrix
        weight_mat = D_inv.dot(weight_mat)
        
    smoothed_data = weight_mat.dot(data)
    #smoothed_data = data.dot(weight_mat.T)
    return smoothed_data;

def load_adjacency_and_distances(surfname="fsaverage.sphere_left", n_hop_iter=10, max_neigh_dist=7):
    #vpair_idxs = np.load(f"data/adjacency_multihop{n_hop_iter}_max{max_neigh_dist}mm_{surfname}.npz")
    #n_verts = 163841;
    #uniq_multi_hop_adj = csr_matrix((np.ones(n_verts), (vpair_idxs[0], vpair_idxs[1])), shape=(n_verts,n_verts))
    uniq_multi_hop_adj = sparse.load_npz(f"data/adjacency_multihop{n_hop_iter}_max{max_neigh_dist}mm_{surfname}.npz")
    dists = np.load(f"data/adjacency_multihop{n_hop_iter}_max{max_neigh_dist}mm_{surfname}.distances.npy")
    return uniq_multi_hop_adj, dists


