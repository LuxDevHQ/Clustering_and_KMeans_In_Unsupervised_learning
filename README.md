
#  Clustering and K-Means in Unsupervised Learning

---

## 1. What is Clustering?

**Clustering** is an **unsupervised learning technique** that automatically finds **natural groupings** within data.

It groups data points such that:

* Points in the **same group** (cluster) are similar to each other.
* Points in **different clusters** are dissimilar.

---

###  Analogy:

> Think of a **box of Lego pieces** all mixed together. You start sorting them by color or size. You’ve just clustered them — without any instructions or labels.

---

## 2. Why is Clustering Important?

Clustering helps:

* Discover hidden **patterns and structures** in data.
* Enable **segmentation** in marketing, healthcare, etc.
* Preprocess data for supervised learning.
* Detect **anomalies or outliers**.

---

## 3. Common Clustering Algorithms

| Algorithm                     | Description                                        |
| ----------------------------- | -------------------------------------------------- |
| K-Means                       | Groups data into K clusters using centroids        |
| Hierarchical                  | Builds a tree of clusters (dendrogram)             |
| DBSCAN                        | Groups based on density; good for irregular shapes |
| Gaussian Mixture Models (GMM) | Soft clustering using probability                  |

---

## 4. K-Means Clustering – In Depth

### 4.1 What is K-Means?

**K-Means** is a **centroid-based algorithm** that partitions data into **K clusters**, where each cluster is defined by its **centroid (center point)**.

---

### 🔍 Analogy: City Water Tanks

> Imagine a city planning to install **K water tanks** to serve all neighborhoods. The goal is to place the tanks so that every house gets water from the nearest tank. K-Means finds the **optimal locations** (centroids) for the tanks.

---

### 4.2 Steps of the K-Means Algorithm

1. **Choose K** (number of clusters)
2. **Randomly initialize K centroids**
3. **Assign each point** to the nearest centroid
4. **Update** the centroid by computing the mean of points in that cluster
5. Repeat steps 3–4 until centroids **don’t move much** (converge)

---

## 5. Distance Metrics in K-Means

Distance is how **"close"** or **"similar"** a data point is to a centroid. K-Means uses **distance to assign points to the nearest centroid**.

###  1. Euclidean Distance (Default)

$$
d(p, q) = \sqrt{(p_1 - q_1)^2 + (p_2 - q_2)^2 + \dots + (p_n - q_n)^2}
$$

> Think of the **straight-line distance** between two points on a map.

---

###  2. Manhattan Distance (L1 norm)

$$
d(p, q) = |p_1 - q_1| + |p_2 - q_2| + \dots + |p_n - q_n|
$$

> Like driving through a city grid — you can’t cut through buildings.

---

###  3. Cosine Distance (used in high-dimensional text)

$$
\text{Cosine Similarity} = \frac{A \cdot B}{\|A\| \|B\|}
$$

> Measures **angle** between two vectors — good for **text and documents**.

---

## 6. How Are Centroids Computed?

Centroids are the **representative points** of clusters. In K-Means, they are usually computed as:

### 1. **Mean (Arithmetic Average)** – *Used in standard K-Means*

$$
\text{Centroid} = \frac{1}{n} \sum_{i=1}^n x_i
$$

> The average of all points in the cluster.

---

###  2. **Median (for robustness)**

* Some variants use the **median** to reduce the effect of outliers.

---

###  3. **Medoid (used in K-Medoids algorithm)**

* The **actual data point** in the cluster with the **minimum average distance** to all other points in that cluster.

---

## 7. Elbow Method – Choosing the Best K

How do we know how many clusters (K) to choose?

###  The Elbow Method

1. Run K-Means with a range of values for K (e.g., 1 to 10)
2. For each K, compute **Inertia** (sum of squared distances to the nearest centroid)
3. Plot K vs Inertia
4. Look for the **"elbow"** point where the curve starts to flatten

> That’s the optimal K — adding more clusters beyond this doesn’t improve the model significantly.

---

###  Elbow Method Code Example

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertias = []
K_range = range(1, 10)

for k in K_range:
    model = KMeans(n_clusters=k)
    model.fit(data)
    inertias.append(model.inertia_)

plt.plot(K_range, inertias, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()
```

---

## 8. K-Means Code: Complete Example

```python
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Sample dataset
data = pd.DataFrame({
    'Age': [25, 30, 45, 35, 50, 23, 40, 60],
    'Income': [30000, 40000, 50000, 45000, 80000, 32000, 60000, 90000]
})

# Apply KMeans
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
data['Cluster'] = kmeans.fit_predict(data)

# Plot clusters
plt.scatter(data['Age'], data['Income'], c=data['Cluster'], cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=200, c='red', label='Centroids')
plt.xlabel('Age')
plt.ylabel('Income')
plt.title('Customer Segmentation')
plt.legend()
plt.show()
```

---

## 9. Variants of K-Means

| Variant               | Description                                             |
| --------------------- | ------------------------------------------------------- |
| **K-Means++**         | Smart centroid initialization (default in scikit-learn) |
| **MiniBatch K-Means** | Faster on large datasets using mini-batches             |
| **K-Medoids**         | Uses medoids (actual data points) instead of means      |
| **Bisecting K-Means** | Divides clusters recursively (hybrid with hierarchical) |

---

## 10. Strengths and Weaknesses

###  Pros

* Simple and fast
* Scales well to large datasets
* Easy to implement

###  Cons

* Must specify **K** in advance
* Sensitive to **initialization**
* Struggles with **non-spherical or overlapping clusters**
* Not good with **categorical data**

---

## 11. Use Cases

| Use Case                   | Description                              |
| -------------------------- | ---------------------------------------- |
| **Customer Segmentation**  | Group users by demographics and behavior |
| **Document Clustering**    | Group similar documents/articles         |
| **Market Basket Analysis** | Cluster shopping patterns                |
| **Image Compression**      | Represent pixels using K colors          |
| **Anomaly Detection**      | Outliers are far from all centroids      |

---

## 12. Summary Table

| Topic                | Key Points                              |
| -------------------- | --------------------------------------- |
| **K-Means**          | Partition-based clustering              |
| **Centroid**         | Mean of the points in the cluster       |
| **Distance Metrics** | Euclidean (default), Manhattan, Cosine  |
| **Initialization**   | Random or K-Means++                     |
| **Evaluation**       | Inertia, Elbow Method, Silhouette Score |
| **Variants**         | MiniBatch, K-Medoids, Bisecting K-Means |

---

## 13. Final Analogy Recap

| Analogy                                | Concept               |
| -------------------------------------- | --------------------- |
| Lego Sorting                           | Clustering            |
| Water Tanks Placement                  | Centroid optimization |
| City Roads                             | Manhattan Distance    |
| Choosing number of tables at a wedding | Elbow Method for K    |

---

##  Complete K-Means + Elbow + Evaluation Code

```python
#  Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

#  Step 1: Create Sample Dataset
data = pd.DataFrame({
    'Age': [25, 30, 45, 35, 50, 23, 40, 60, 29, 48, 33, 55, 37, 41, 58],
    'Income': [30000, 40000, 50000, 45000, 80000, 32000, 60000, 90000,
               39000, 71000, 42000, 85000, 47000, 61000, 92000]
})

#  Step 2: Feature Scaling (important for distance-based algorithms)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Step 3: Elbow Method to Find Optimal K
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(scaled_data)
    inertias.append(kmeans.inertia_)
    score = silhouette_score(scaled_data, kmeans.labels_)
    silhouette_scores.append(score)

# Plot Inertia vs K (Elbow Method)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method - Inertia')

# Plot Silhouette Score vs K
plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, 'go-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for K')
plt.tight_layout()
plt.show()

#  Step 4: Apply KMeans with Optimal K (based on elbow/silhouette)
optimal_k = 3  # You can change this based on the elbow/silhouette
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(scaled_data)
data['Cluster'] = clusters

#  Step 5: Visualize Clusters
plt.figure(figsize=(8, 6))
colors = ['purple', 'orange', 'green', 'blue', 'red']
for i in range(optimal_k):
    plt.scatter(data['Age'][data['Cluster'] == i],
                data['Income'][data['Cluster'] == i],
                label=f'Cluster {i+1}',
                color=colors[i])
    
# Add cluster centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='black', label='Centroids', marker='X')

plt.xlabel("Age")
plt.ylabel("Income")
plt.title(f"Customer Segmentation (K={optimal_k})")
plt.legend()
plt.grid(True)
plt.show()

#  Step 6: Evaluate Clustering Performance
print(f"Final Inertia (Total within-cluster distance): {kmeans.inertia_:.2f}")
sil_score = silhouette_score(scaled_data, clusters)
print(f"Silhouette Score: {sil_score:.4f}")

#  Step 7: Display Cluster Assignments
print("\nCluster Assignments:")
print(data.sort_values(by='Cluster'))
```

---

##  What This Code Covers

| Section                    | Purpose                                     |
| -------------------------- | ------------------------------------------- |
| **Scaling**                | Standardizes features to equal importance   |
| **Elbow Method**           | Finds optimal K using **inertia plot**      |
| **Silhouette Score**       | Measures cluster quality (higher is better) |
| **KMeans Clustering**      | Applies algorithm using best K              |
| **Visualization**          | Shows clusters + centroids                  |
| **Performance Evaluation** | Inertia + Silhouette Score                  |
| **Final Output**           | Clustered data in DataFrame                 |




