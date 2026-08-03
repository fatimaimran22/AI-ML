"""
    a · b = Σ aibi = ‖a‖ ‖b‖ cos(θ)

    a . b = 0 ---------> Vectors are orthogonal, prependicular, cos(90) = 0 -----> unrelated

    cosine similary = (a · b) / ‖a‖ ‖b‖ ------> ranges (-1, 1)

    --> Are Vectors pointing in same direction ?, i.e they are similar e.g I love cats, Cats are wonderful pets ---> both are similar

    | Angle | Cosine Similarity | Meaning                    |
    | ----- | ------------------| -------------------------- |
    | 0°    |     1             | Exactly the same direction |
    | 45°   |     0.71          | Quite similar              |
    | 90°   |     0             | Unrelated (perpendicular)  |
    | 180°  |     -1            | Completely opposite        |

"""

import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


a = np.array([1, 2])
b = np.array([6, 9])

print(f"Cosine Similarity between two vectors is : {cosine_similarity(a, b):.4f}")