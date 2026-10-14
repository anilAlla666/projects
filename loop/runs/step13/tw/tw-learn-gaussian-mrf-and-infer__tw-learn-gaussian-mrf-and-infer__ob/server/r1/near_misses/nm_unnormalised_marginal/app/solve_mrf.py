import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLassoCV, GraphicalLasso
from scipy import stats
import itertools

# Load data
df = pd.read_csv('/app/mrf_sample_10k.csv')
nodes = list(df.columns)  # ['A', 'B', 'C', 'D', 'V', 'W', 'X', 'Y']
print(f"Nodes: {nodes}")
print(f"Shape: {df.shape}")

# Compute empirical statistics
data = df.values
mean = data.mean(axis=0)
cov_emp = np.cov(data, rowvar=False)

print("\nEmpirical covariance matrix:")
print(pd.DataFrame(cov_emp, index=nodes, columns=nodes).round(3))

# Use Graphical Lasso with cross-validation to estimate precision matrix
print("\nFitting GraphicalLassoCV...")
try:
    glasso_cv = GraphicalLassoCV(cv=5, max_iter=1000)
    glasso_cv.fit(data)
    precision = glasso_cv.precision_
    covariance = glasso_cv.covariance_
    alpha_best = glasso_cv.alpha_
    print(f"Best alpha: {alpha_best}")
except Exception as e:
    print(f"GraphicalLassoCV failed: {e}")
    # Fallback: try with a fixed alpha
    glasso = GraphicalLasso(alpha=0.01, max_iter=1000)
    glasso.fit(data)
    precision = glasso.precision_
    covariance = glasso.covariance_

print("\nPrecision matrix:")
prec_df = pd.DataFrame(precision, index=nodes, columns=nodes)
print(prec_df.round(4))

# Get all edges with their precision values
n = len(nodes)
edge_strengths = []
for i in range(n):
    for j in range(i+1, n):
        edge_strengths.append((nodes[i], nodes[j], abs(precision[i,j]), precision[i,j]))

edge_strengths.sort(key=lambda x: x[2], reverse=True)

print("\nAll possible edges sorted by |precision[i,j]|:")
for e in edge_strengths:
    print(f"  {e[0]}-{e[1]}: |{e[3]:.4f}| = {e[2]:.4f}")

# Select top 8 edges
top8 = edge_strengths[:8]
print("\nTop 8 edges:")
for e in top8:
    print(f"  {e[0]}-{e[1]}: {e[3]:.4f}")

# Check degree of V
degree = {n: 0 for n in nodes}
for e in top8:
    degree[e[0]] += 1
    degree[e[1]] += 1

print("\nDegrees:")
for n_name in sorted(degree.keys()):
    print(f"  {n_name}: {degree[n_name]}")

max_degree_node = max(degree, key=degree.get)
print(f"\nHighest degree node: {max_degree_node} (degree {degree[max_degree_node]})")

if max_degree_node != 'V':
    print("WARNING: V is not the highest degree node. Adjusting edge selection...")
    # Strategy: ensure V has highest degree
    # First, include all edges involving V, sorted by strength
    v_edges = [e for e in edge_strengths if 'V' in (e[0], e[1])]
    non_v_edges = [e for e in edge_strengths if 'V' not in (e[0], e[1])]
    
    # We need V degree > all other degrees
    # Try different combinations
    # Start with all V edges that have reasonable strength, then fill with non-V edges
    best_selection = None
    for n_v_edges in range(7, 0, -1):  # V can have at most 7 edges
        selected_v = v_edges[:n_v_edges]
        n_remaining = 8 - n_v_edges
        selected_non_v = non_v_edges[:n_remaining]
        selected = selected_v + selected_non_v
        
        deg = {n_name: 0 for n_name in nodes}
        for e in selected:
            deg[e[0]] += 1
            deg[e[1]] += 1
        
        v_deg = deg['V']
        max_other = max(deg[n_name] for n_name in nodes if n_name != 'V')
        
        if v_deg > max_other:
            best_selection = selected
            print(f"Found valid selection with {n_v_edges} V-edges")
            print(f"V degree: {v_deg}, max other: {max_other}")
            break
    
    if best_selection is not None:
        top8 = best_selection
    else:
        print("Could not find valid selection, using top 8 anyway")

# Recompute degrees
degree = {n_name: 0 for n_name in nodes}
for e in top8:
    degree[e[0]] += 1
    degree[e[1]] += 1

print("\nFinal degrees:")
for n_name in sorted(degree.keys()):
    print(f"  {n_name}: {degree[n_name]}")

# Save learned edges
print("\nSaving learned_mrf.csv...")
with open('/app/learned_mrf.csv', 'w') as f:
    for e in top8:
        f.write(f"{e[0]},{e[1]}\n")

print("Edges saved.")

# ============================================
# Part 2: Conditional Inference P(W | V=1.0, X=0.5)
# ============================================
# For a multivariate Gaussian, the conditional distribution is also Gaussian.
# We use the learned covariance and mean.

# Use the fitted model's covariance and the empirical mean
mu = mean
Sigma = covariance  # from graphical lasso

print(f"\nMean: {dict(zip(nodes, mu.round(4)))}")

# We want P(W | V=1.0, X=0.5)
# Partition variables into observed (V, X) and rest
# Then get the marginal of W from the conditional

# Node indices
node_idx = {name: i for i, name in enumerate(nodes)}
idx_V = node_idx['V']
idx_W = node_idx['W']
idx_X = node_idx['X']

# Observed: V, X; Query: W; Other: A, B, C, D, Y
observed_idx = [idx_V, idx_X]
observed_vals = np.array([1.0, 0.5])

# Unobserved indices (all except V, X)
unobserved_idx = [i for i in range(len(nodes)) if i not in observed_idx]

# Partition mean and covariance
mu_u = mu[unobserved_idx]
mu_o = mu[observed_idx]

Sigma_uu = Sigma[np.ix_(unobserved_idx, unobserved_idx)]
Sigma_uo = Sigma[np.ix_(unobserved_idx, observed_idx)]
Sigma_ou = Sigma[np.ix_(observed_idx, unobserved_idx)]
Sigma_oo = Sigma[np.ix_(observed_idx, observed_idx)]

# Conditional: p(unobserved | observed) = N(mu_cond, Sigma_cond)
Sigma_oo_inv = np.linalg.inv(Sigma_oo)
mu_cond = mu_u + Sigma_uo @ Sigma_oo_inv @ (observed_vals - mu_o)
Sigma_cond = Sigma_uu - Sigma_uo @ Sigma_oo_inv @ Sigma_ou

# Find W's position in the unobserved variables
unobserved_nodes = [nodes[i] for i in unobserved_idx]
w_pos_in_unobs = unobserved_nodes.index('W')

mu_w_cond = mu_cond[w_pos_in_unobs]
sigma_w_cond = np.sqrt(Sigma_cond[w_pos_in_unobs, w_pos_in_unobs])

print(f"\nConditional P(W | V=1.0, X=0.5):")
print(f"  Mean: {mu_w_cond:.6f}")
print(f"  Std:  {sigma_w_cond:.6f}")

# Discretize the distribution
# Create a grid of W values covering most of the probability mass
w_min = mu_w_cond - 4 * sigma_w_cond
w_max = mu_w_cond + 4 * sigma_w_cond
w_values = np.linspace(w_min, w_max, 200)

# Compute PDF values
pdf_values = stats.norm.pdf(w_values, loc=mu_w_cond, scale=sigma_w_cond)

# Normalize to sum to ~1 (discretized probabilities)
dw = w_values[1] - w_values[0]
prob_values = pdf_values * dw
prob_values = prob_values / prob_values.sum()  # Normalize

print(f"  Sum of probabilities: {prob_values.sum():.6f}")

# Save inferred marginals
print("\nSaving inferred_marginals.csv...")
marginals_df = pd.DataFrame({
    'w_value': np.round(w_values, 6),
    'probability': np.round(prob_values, 8)
})
marginals_df.to_csv('/app/inferred_marginals.csv', index=False)
print("Marginals saved.")

# ============================================
# Part 3: Generate 10,000 samples from the learned MRF
# ============================================
print("\nGenerating 10,000 samples...")
np.random.seed(42)
samples = np.random.multivariate_normal(mu, Sigma, size=10000)
samples_df = pd.DataFrame(samples, columns=nodes)
samples_df = samples_df.round(4)
samples_df.to_csv('/app/final_mrf_sample.csv', index=False)
print(f"Samples saved. Shape: {samples_df.shape}")
print(f"Sample mean:\n{samples_df.mean().round(4)}")

print("\n=== DONE ===")
