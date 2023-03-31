import os
import glob
import numpy as np
import torch.utils.data as data
import os.path
import torch
from plyfile import PlyData

input_file = 'COMP0016_App/inference_module/model/All_teeth'

class JawDataset(data.Dataset):
    def __init__(self,
                 root=input_file,
                 npoints=2048,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation

        self.points = []
        self.labels = []

        self.files = glob.glob(os.path.join(input_file, "All_teeth/{}/inputs/*".format(self.split)))
        self.files = sorted(self.files)

        count = 0

        for file in self.files:
            with open(file, 'rb') as f:
                    plydata = PlyData.read(f)
            pts = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
            sets = len(pts) // 2048

            file_name = os.path.basename(file)
            grade = file_name.split(' ')[0]
            max_grade = 0
            if len(grade) == 4:
                grades = [i for i in grade]
                max_grade = max(grades)
            elif len(grade) == 5:
                grades = [i for i in grade[1:]]
                max_grade = max(grades)

            for i in range(sets):
                num_points = np.array(len(pts))
                choice = np.random.choice(num_points, self.npoints, replace=True)
                num_points = np.delete(num_points, np.where(np.isin(num_points, choice)))
                point_set = pts[choice, :]
                point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
                dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
                point_set = point_set / dist  # scale

                if self.data_augmentation:
                    if max_grade == '1':
                        for i in range(3):
                            # random rotation
                            theta = np.random.uniform(0, np.pi * 2)
                            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  
                            # random jitter
                            point_set += np.random.normal(0, 0.02, size=point_set.shape)
                            point_set = torch.from_numpy(point_set.astype(np.float32))
                            self.points.append(point_set)
                            self.labels.append(max_grade)
                            point_set = point_set.numpy()
                    else:
                        theta = np.random.uniform(0, np.pi * 2)
                        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                        # random rotation
                        point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  
                        # random jitter
                        point_set += np.random.normal(0, 0.02, size=point_set.shape)
                        point_set = torch.from_numpy(point_set.astype(np.float32))
                        self.points.append(point_set)
                        self.labels.append(max_grade)
                else:
                    point_set = torch.from_numpy(point_set.astype(np.float32))
                    self.points.append(point_set)
                    self.labels.append(max_grade)
            count += 1
            if count % 10 == 0:
                print(file)


    def __getitem__(self, index):
        point = self.points[index]
        label = self.labels[index]
        label = torch.from_numpy(np.array([label]).astype((np.float32)))
        return point, label


    def __len__(self):
        return len(self.points)
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pytorch3d
    from pytorch3d.structures import Pointclouds

    d = JawDataset(input_file)
    print(len(d))
    print(d[0][0], d[0][1])
    print(d[0][0].dim())
    
    print(d[0][0])
    print(d[0][0].shape)

    pc = Pointclouds([d[0][0]])
    np_points = pc.points_packed().detach().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(np_points[:, 0], np_points[:, 1], np_points[:, 2])
    plt.show()