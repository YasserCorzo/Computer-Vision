import cv2
import numpy as np

from submission import *

# rotation vector
r = np.array([1, 0, 0]) * np.pi/2 # 90 degrees rotation around x-axis
R, _ = cv2.Rodrigues(r)

R_imp = rodrigues(r.reshape(-1, 1))

r, _ = cv2.Rodrigues(R_imp)
r_imp = invRodrigues(R_imp)

print(r)
print(r_imp)
