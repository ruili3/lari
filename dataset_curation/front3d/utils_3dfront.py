'''
helper functions to make LDIs for 3D-FRONT dataset
'''
import os
import numpy as np

def invert_transformation_metrix(matrix):
    matrix = np.array(matrix)
    return np.linalg.inv(matrix)


def save_matrix(matrix, cam_len, path, id):
    '''
    save the tranformation matrix into a 3x4 numpy array (aligned with Objaverse)
    '''
    data = {
        "T_b_w2cam": matrix[:3, :4], 
        "cam_len": cam_len
    }

    RT_path = os.path.join(path,"{:03d}.npy".format(id))
    np.save(RT_path, data)