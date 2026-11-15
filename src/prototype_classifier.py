import numpy as np


class PrototypeClassifier:

    def __init__(self):
        '''
        Create an empty mapping of class labels to prototypes.
        '''
        self.prototypes = {}

    def fit(self, embeddings, labels):
        '''
        Replace prototypes with class means of raw support embeddings.
        Reject empty inputs, invalid vectors and missing labels.
        '''
        embeddings = np.asarray(embeddings, dtype=float)
        labels = np.asarray(labels)
        if (embeddings.ndim != 2 or 0 in embeddings.shape or labels.ndim != 1
                or len(labels) != len(embeddings)):
            raise ValueError("Expected a nonempty embedding matrix and one label per row")
        if (not np.isfinite(embeddings).all()
                or np.any(np.linalg.norm(embeddings, axis=1) == 0)):
            raise ValueError("Support vectors must be finite and nonzero")
        if any(not isinstance(label, str) or not label.strip() for label in labels):
            raise ValueError("Labels must be nonempty strings")
        prototypes = {str(label): embeddings[labels == label].mean(axis=0)
                      for label in np.unique(labels)}
        if any(np.linalg.norm(vector) == 0 for vector in prototypes.values()):
            raise ValueError("A class mean has zero norm")
        self.prototypes = prototypes

    def predict(self, embedding, top_k=3):
        '''
        Return classes ranked by cosine similarity, not probabilities.
        '''
        if not self.prototypes:
            raise ValueError("Fit the classifier before prediction")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        embedding = np.asarray(embedding, dtype=float)
        shape = next(iter(self.prototypes.values())).shape
        if (embedding.shape != shape or not np.isfinite(embedding).all()
                or np.linalg.norm(embedding) == 0):
            raise ValueError("Query vector must match the prototypes and be finite and nonzero")
        scores = []
        for label, prototype in self.prototypes.items():
            similarity = np.dot(embedding, prototype) / (
                np.linalg.norm(embedding) * np.linalg.norm(prototype))
            scores.append({"action": label, "score": float(similarity)})
        return sorted(scores, key=lambda item: item["score"], reverse=True)[:top_k]
