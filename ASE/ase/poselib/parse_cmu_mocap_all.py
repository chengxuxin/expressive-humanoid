import pandas as pd
import matplotlib.pyplot as plt
from ruamel.yaml import YAML
import numpy as np

path = "./data/configs/"
def find_entries(word, column):
    subset = column[column.str.contains(word, na=False, case=False)]
    num_entries = len(subset)
    num_unique_entries = subset.nunique()
    indices = subset.index.tolist()
    unique_entries = subset.unique()
    print(f"Found {num_entries} entries containing '{word}', {num_unique_entries} unique entries.")
    return num_entries, num_unique_entries, indices, unique_entries

def save_yaml(motion_ids, motion_descriptions, name):
    yaml = YAML()
    yaml.preserve_quotes = True  # Preserve quotes if necessary
    motions_data = {"root": "../retarget_npy/"}
    for i, motion_id in enumerate(motion_ids):
        descrp = motion_descriptions[i]
        descrp = descrp.replace("/", ", ").replace("\\", ", ")
        motions_data[motion_id] = {
            "trim_beg": -1,  # Assuming these values are constant, modify as needed
            "trim_end": -1,
            "weight": 1.0,
            "description": f"{descrp}",
            "difficulty": 4  # Modify as needed
        }
    # Constructing the final dictionary structure
    final_structure = {
        "motions": motions_data
    }
    # Writing to a YAML file
    with open(path + f'motions_autogen_debug_{name}.yaml', 'w') as file:
        yaml.dump(final_structure, file)

    print('YAML file created with the specified structure.')

file_path = path + "cmu-mocap-index-spreadsheet.xls"
data = pd.read_excel(file_path)
description_column = data.iloc[:, 1]
motion_id_column = data.iloc[:, 0]

forbbiden_words = ["ladder", "suitcase", "uneven", "terrain", "stair", "stairway", "stairwell", "clean", "box", "climb", "backflip", "handstand", "sit", "hang"]
# target_words = ["walk", "run", "jump", "navigate", "basketball", "dance", "punch", "fight", "push", "pull", "throw", "catch", "crawl", "wave", "high five", "hug", "drink", "wash", "signal", "balance", "strech", "leg", "bend", "squat", "traffic", "high-five", "low-five"]
target_words = ["walk", "navigate", "basketball", "dance", "punch", "fight", "push", "pull", "throw", "catch", "crawl", "wave", "high five", "hug", "drink", "wash", "signal", "balance", "strech", "leg", "bend", "squat", "traffic", "high-five", "low-five"]
# target_words = ["walk", "navigate"]
target_results = []
for word in target_words:
    target_results.append(find_entries(word, description_column))

print("\n Searching for forbidden words:")
fbd_indices = []
for word in forbbiden_words:
    fbd_indices.extend(find_entries(word, description_column)[2])
print(f"Found {len(fbd_indices)} forbidden entries.")
fbd_indices = list(set(fbd_indices))
print(f"Found {len(fbd_indices)} unique forbidden entries.")

print("\n Filtering forbidden words:")
indices_all = []
for i, result in enumerate(target_results):
    indices = result[2]
    filtered_indices = [index for index in indices if index not in fbd_indices]
    filtered_entries = description_column[filtered_indices]
    filtered_unique_entries = list(set(filtered_entries))
    print(f"Found {len(filtered_indices)} entries for '{target_words[i]}' after filtering, {len(filtered_unique_entries)} unique entries.")

    motion_ids = motion_id_column[filtered_indices]
    indices_all.extend(filtered_indices)
    
    if target_words[i] in ["walk", "dance", "basketball", "punch"]:
        save_yaml(motion_ids, filtered_entries.tolist(), target_words[i])

indices_all_unique = list(set(indices_all))
motion_ids_all_unique = motion_id_column[indices_all_unique]
save_yaml(motion_ids_all_unique, description_column[indices_all_unique].tolist(), "all_no_run_jump")


# exit()

text_all = description_column[indices_all_unique].tolist()
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')
embeddings = model.encode(text_all)
print(embeddings.shape)

from sklearn.cluster import KMeans
k = 8  # Example number of clusters
kmeans = KMeans(n_clusters=k, random_state=0).fit(embeddings)

# Get cluster labels for each point in your dataset
labels = kmeans.labels_
np.save("labels.npy", labels)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# Reduce dimensions (e.g., to 2D)
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# Plot
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('2D Visualization of Embeddings Clusters')
# plt.legend([0, 1, 2, 3, 4])
plt.show()
# plt.savefig("cluster.png")

# get all text for label 0
for i_cluster in range(k):
    indices_label_0 = [i for i, label in enumerate(labels) if label == i_cluster]
    print([text_all[i] for i in indices_label_0], "\n")