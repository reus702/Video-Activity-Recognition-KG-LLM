import numpy as np


class PrototypeClassifier:

    def __init__(self):
        self.prototypes = {}

    def fit(self, embeddings, labels):

        embeddings = np.asarray(embeddings)
        labels = np.asarray(labels)

        for label in np.unique(labels):

            vectors = embeddings[labels == label]

            self.prototypes[str(label)] = vectors.mean(
                axis=0
            )

    def predict(self, embedding, top_k=3):

        scores = []

        for label, prototype in self.prototypes.items():

            similarity = np.dot(embedding, prototype) / (
                np.linalg.norm(embedding)
                * np.linalg.norm(prototype)
                + 1e-8
            )

            scores.append({
                "action": label,
                "score": float(similarity)
            })

        return sorted(
            scores,
            key=lambda x: x["score"],
            reverse=True
        )[:top_k]