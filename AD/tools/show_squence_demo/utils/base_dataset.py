import copy
import numpy as np

from collections import defaultdict

from .components import Object3D

class DataCollect:
    def __init__(self, name='Waymo', 
                 color_attr=[], 
                 text_attr=[],
                 show_text=False):
        # super().__init__(name=name)
        self.name = name
        self.num_classes = 3

        self.datas = list()
        self.data_labels = list()
        self.labels = list()
        self.idx_names = list()

        self.label_to_names = {}

        self.color_attr = color_attr
        self.text_attr = text_attr
        self.show_text = show_text

    def offline_process_infos(self, **infos):
        self.datas.clear()
        self.labels.clear()
        self.data_labels.clear()

        infos_keys = infos.keys()

        if "idx_names" not in infos_keys or "pts" not in infos_keys:
            raise ValueError("Need idx_names' or pts' infos")

        pts_len = len(infos["pts"])
        idx_len = len(infos["idx_names"])
        assert pts_len == idx_len, f"length of pts != idx_names"

        names = dict()

        for idx in range(pts_len):

            pts = infos["pts"][idx]
            pts.astype(np.float32)
            self.datas.append(pts)
            self.data_labels.append(infos['pts_label'][idx])

            idx_n = infos["idx_names"][idx]
            self.idx_names.append(idx_n)

            label_info = defaultdict(dict)
            for key in infos_keys:
                if key == "idx_names" or "pts" in key:
                    continue

                if key not in names.keys():
                    names[key] = set()

                bbox = infos[key][idx]["bbox"]
                bbox_len = len(bbox)
                repeat_name = np.repeat([key], bbox_len)

                label_info[key]["name"] = repeat_name
                label_info[key]["bbox"] = bbox

                
                meta_center = copy.deepcopy(bbox[:, :3])
                label_info[key]["meta_center"] = meta_center

                # @todo: other features
                if "id" in infos[key][idx].keys():
                    label_info[key]["id"] = infos[key][idx]["id"]
                    if "id" in self.color_attr:
                        names[key].update(label_info[key]["id"])
                if "class" in infos[key][idx].keys():
                    label_info[key]["class"] = infos[key][idx]["class"]
                    if "class" in self.color_attr:
                        names[key].update(label_info[key]["class"])
                if "score" in infos[key][idx].keys():
                    label_info[key]["score"] = infos[key][idx]["score"]

            self.labels.append(label_info)

        self.label_to_names = self.get_label_to_names(names)

    def get_label_to_names(self, names):
        """Returns a label to names dictonary object.

        Returns:
            A dict where keys are label numbers and
            values are the corresponding names.
        """
        if len(self.color_attr) == 0:
            return dict.fromkeys(names.keys(), list())
        
        new_names = dict()
        for key, val in names.items():
            if len(val) == 0:
                new_names[key] = []
            for sub_name in val:
                new_name = key+"_"+str(sub_name)
                new_names[new_name] = []
        return new_names

    def is_tested(self, attr):
        """Checks whether a datum has been tested.

        Args:
            attr: The attributes associated with the datum.

        Returns:
            This returns True if the test result has been stored for the datum with the
            specified attribute; else returns False.
        """
        return False

    def save_test_result(self, results, attr):
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the attribute passed.
            attr: The attributes that correspond to the outputs passed in results.
        """
        return

    # @staticmethod
    # def read_lidar(path):
    #     """Reads lidar data from the path provided.

    #     Returns:
    #         A data object with lidar information.
    #     """
    #     assert Path(path).exists()

    #     return np.fromfile(path, dtype=np.float32).reshape(-1, 6)name_ns
    def read_label(self, labels):
        """Reads labels of bound boxes.

        Returns:
            The data objects with bound boxes information.
        """
        objects = []
        names = labels.keys()
        for name in names:
            attr_keys = labels[name].keys()

            name_ns = labels[name]["name"]
            bboxs = labels[name]["bbox"]
            meta_centers = labels[name]["meta_center"]

            bboxs_len = len(bboxs)
            for i in range(bboxs_len):
                center = [float(bboxs[i][0]), float(
                    bboxs[i][1]), float(bboxs[i][2])]
                size = [float(bboxs[i][4]), float(
                    bboxs[i][5]), float(bboxs[i][3])]
                heading = float(bboxs[i][6])
                meta_center = [float(meta_centers[i][0]), float(
                    meta_centers[i][1]), float(meta_centers[i][2])]

                cls = labels[name]["class"][i] if "class" in attr_keys else ""
                score = labels[name]["score"][i] if "score" in attr_keys else 1.
                id = labels[name]["id"][i] if "id" in attr_keys else "" 

                show_name = name_ns[i]
                if "class" in self.color_attr:
                    show_name = name_ns[i] + "_"+ cls
                elif "id" in self.color_attr and id!= "":
                    show_name = name_ns[i] + "_"+ str(id)

                text = ""
                if "name" in self.text_attr:
                    text = text + " " + name_ns[i]
                if "class" in self.text_attr:
                    text = text + " " + cls
                if "score" in self.text_attr:
                    text = text + " " + f"{score:.2f}"
                if "id" in self.text_attr:
                    text = text + " " + str(id)
                text = text.strip()

                show_text=self.show_text
                if text == "":
                    show_text = False
                
                objects.append(
                    Object3D(center=center, 
                             size=size, 
                             yaw=heading, 
                             name=show_name,
                             cls=cls,
                             score=score,
                             id=id,
                             text=text,
                             show_meta=show_text,
                             meta_center=meta_center,
                             show_arrow=True))

        return objects

    def get_split_list(self):
        """Returns the list of data splits available.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split objeprefix
            ValueError: Indicates that the sget_label_to_namesplit name passed is incorrect. The
            split name should be one of 'training', 'test', 'validation', or
            'all'.
        """
        spilt_list = []
        for id in range(len(self.datas)):
            data_dict = {'data': self.datas[id], 
                         'label': self.labels[id], 
                         'data_label': self.data_labels[id],
                        }
            spilt_list.append(data_dict)
        return spilt_list

    def __len__(self):
        return len(self.datas)

    def get_split(self, prefix):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return DataSplit(self, self.idx_names, prefix)


class DataSplit():
    def __init__(self, dataset, idx_names, prefix=""):

        self.idx_names = idx_names
        self.data_list = dataset.get_split_list()
        self.dataset = dataset
        self.prefix = prefix

    def __len__(self):
        return len(self.data_list)

    def get_data(self, idx):
        data_dict = self.data_list[idx]

        pts = data_dict['data']
        label = self.dataset.read_label(data_dict['label'])
        pts_label = data_dict['data_label']

        data = {
            'point': pts,
            'feat': None,
            'bounding_boxes': label,
            'pts_label': pts_label,
        }
        return data

    def get_attr(self, idx):

        attr = {'name': self.prefix+":"+self.idx_names[idx]}
        return attr