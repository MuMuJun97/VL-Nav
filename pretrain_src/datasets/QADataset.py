import json
import h5py
import argparse
import numpy as np
from tqdm import tqdm
import ast
from copy import deepcopy
import random
import yaml
from pathlib import Path
from easydict import EasyDict

class BaseDataset(object):
    def __init__(self,config,split,in_memory=True):
        self.split = split
        self.config = config
        self.data_dir = Path(config.DATA_DIR).resolve()
        self.soon_file = self.data_dir / self.config.SOON_DIR / self.config.SOON_SPLIT[split]
        self.fr2r_file = self.data_dir / self.config.FR2R_DIR / self.config.FR2R_SPLIT[split]

        self.soon_data = []
        self.preprocess_soon()

        self.fr2r_data = []
        self.preprocess_fr2r()

        self.data = self.soon_data + self.fr2r_data

        # read image features
        self.in_memory = in_memory
        if self.in_memory:
            self._feature_store = {}
        self.img_ft_file = self.data_dir / self.config.Img_Features_File_Map

        if config.With_Object_Feats:
            self.obj_ft_file = self.data_dir / self.config.Object_Features_File_Map
        else:
            self.obj_ft_file = None


    def __len__(self):
        return len(self.soon_data) + len(self.fr2r_data)

    def preprocess_soon(self):
        assert self.soon_file.exists()
        with open(str(self.soon_file),"r") as f:
            soon_data = json.load(f)

        promptQAs = [
            "What are the attributes of the target object?",
            "What is the relationship between the target object and other objects in the room?",
            "Which room or area is the target object in?",
            "What is the relationship between the target room and other neighboring rooms?",
            "What is the navigation instruction for the current scene?",
            "What is the target object of navigation?",
        ]

        pbar = tqdm(soon_data,desc="preprocess soon data:")
        for idx,_ in enumerate(pbar):
            for path in soon_data[idx]['path']:
                for instr in soon_data[idx]['instructions']:
                    item = dict()

                    item['path'] = path
                    item['instruction'] = deepcopy(instr)
                    valid_bbox = []
                    for bbox in soon_data[idx]['bboxes']:
                        if bbox['image_id'] == path[-1]:
                            if bbox['obj_name'] is None:
                                bbox['obj_name'] = 'None'
                            valid_bbox.append(bbox)
                    item['bbox'] = random.choice(valid_bbox)
                    item['scan'] = item['bbox']['scan']
                    item['instruction'].insert(-1, item['bbox']['obj_name'])
                    self.soon_data.append(item)
            pbar.update(1)

        soon_lens = len(self.soon_data) // 10
        pbar = tqdm(self.soon_data, desc="generate soon qa:")
        for idx,_ in enumerate(pbar):
            # for qa:
            ridx = random.randint(0, 5)
            if ridx == 4:
                question = "Question: {}\nAnswer:".format(
                    promptQAs[ridx]
                )
                answer = "{}".format(self.soon_data[idx]['instruction'][4])
                self.soon_data[idx]['qa'] = dict()
                self.soon_data[idx]['qa']['question'] = question
                self.soon_data[idx]['qa']['answer'] = answer
            else:
                instr_idx = list(range(soon_lens))
                choices = random.choices(instr_idx, k=2)
                options = ['A', 'B', 'C']
                choices_txt = list()
                choices_txt.append(" ".join(self.soon_data[idx]['instruction'][ridx].split()))
                choices_txt.append(
                    " ".join(self.soon_data[choices[0]*10]['instruction'][ridx].split())
                )
                choices_txt.append(
                    " ".join(self.soon_data[choices[1]*10]['instruction'][ridx].split())
                )
                random_choices = [0, 1, 2]
                random.shuffle(random_choices)
                choices_txt = [choices_txt[ri] for ri in random_choices]
                answer_option = random_choices.index(0)
                answer_option = options[answer_option]
                answer_option = "(" + answer_option + ")"

                choice_list = []
                for i, c in enumerate(choices_txt):
                    choice_list.append("({}) {}".format(options[i], c))
                choice_txt = " ".join(choice_list)

                question = "Question: {}\nOptions: {}\nAnswer:".format(promptQAs[ridx],choice_txt)
                answer = "The answer is {}".format(answer_option)

                self.soon_data[idx]['qa'] = dict()
                self.soon_data[idx]['qa']['question'] = question
                self.soon_data[idx]['qa']['answer'] = answer
            pbar.update(1)

    def preprocess_fr2r(self):
        assert self.fr2r_file.exists()
        with open(str(self.fr2r_file),"r") as f:
            fr2r_data = json.load(f)
        pbar = tqdm(fr2r_data, desc="preprocess fine-grained data:")
        for idx, _ in enumerate(pbar):
            for j,chunk in enumerate(fr2r_data[idx]['chunk_view']):
                for k,sub_path in enumerate(chunk):
                    item = dict()
                    item['scan'] = fr2r_data[idx]['scan']
                    item['fr2r'] = {
                        'distance': fr2r_data[idx]['distance'],
                        'path_id': fr2r_data[idx]['path_id'],
                        'heading': fr2r_data[idx]['heading'],
                    }
                    start_index = sub_path[0]-1
                    end_index = sub_path[1]
                    item['path'] = fr2r_data[idx]['path'][start_index:end_index]
                    new_instructions = ast.literal_eval(fr2r_data[idx]['new_instructions'])
                    item['instructions'] = {
                        'full': fr2r_data[idx]['instructions'][j],
                        'sub_instr': " ".join(new_instructions[j][k])
                    }
                    self.fr2r_data.append(item)

    def get_scan_viewpoint_feature(self, scan, viewpoint):
        """
        Args:
            scan: matterport 3d scene
            viewpoint: the prefix/name of current node/viewpoint
        Returns:
            view_fts: [num_views, img_feat_dim] [36,1768]
                num_views=36(or 37), img_feat_dim=image_feat_size(0:768)+image_prob_size(768:1768)
            obj_fts: [num_objects, obj_feat_dim] [num_objects,2048+1601=3649]
                obj_feat_dim=obj_feat_size(0:2048)+obj_prob_size(2048:)
            obj_attrs:
                'bboxes': [num_objects,4]
                'directions': [num_objects,2]
                'obj_ids': [num_objects,]
                'sizes': [num_objects,2]
        """
        key = '%s_%s' % (scan, viewpoint)
        if self.in_memory and key in self._feature_store:
            view_fts, obj_fts, obj_attrs = self._feature_store[key]
        else:
            with h5py.File(str(self.img_ft_file), 'r') as f:
                view_fts = f[key][...].astype(np.float32)
                view_fts = view_fts[:36]

            obj_attrs = {}
            obj_ft_lens = self.config.obj_feat_size+self.config.obj_prob_size
            obj_fts = np.zeros((0, obj_ft_lens), dtype=np.float32)

            if self.obj_ft_file is not None:
                with h5py.File(self.obj_ft_file, 'r') as f:
                    if key in f:
                        obj_fts = f[key][...].astype(np.float32)
                        obj_fts = obj_fts[:self.config.max_objects]
                        for attr_key, attr_value in f[key].attrs.items():
                            if attr_key in ['directions', 'bboxes', 'obj_ids']:
                                obj_attrs[attr_key] = attr_value[:self.config.max_objects]
                        obj_attrs['bboxes'] = np.array(obj_attrs['bboxes']).astype(np.float32)
                        obj_attrs['sizes'] = np.zeros((len(obj_attrs['bboxes']), 2), dtype=np.float32)
                        obj_attrs['sizes'][:, 0] = obj_attrs['bboxes'][:, 2] - obj_attrs['bboxes'][:, 0]
                        obj_attrs['sizes'][:, 1] = obj_attrs['bboxes'][:, 3] - obj_attrs['bboxes'][:, 1]
            if self.in_memory:
                self._feature_store[key] = (view_fts, obj_fts, obj_attrs)
        return view_fts, obj_fts, obj_attrs

    def __getitem__(self, index):
        if index >= len(self.soon_data):
            # for fine-grained dataset: sub-path <-> sub-instruction
            item = self.data[index]
            scan = item['scan']
            viewpoint = item['path'][-1]
            view_fts, obj_img_fts, obj_attrs = self.get_scan_viewpoint_feature(scan, viewpoint)
            question = "Full Instruction: {}".format(" ".join(item['instructions']['full'].split()))
            answer = "Sub-Instruction is {}".format(" ".join(item['instructions']['sub_instr'].split()))
            data_dict = {
                'question': question,
                'answer': answer,
                'instruction': " ".join(item['instructions']['full'].split()),
                'img_feats': view_fts[:, :self.config.image_feat_size],
                'obj_feats': obj_img_fts[:, :self.config.obj_feat_size],
            }
            return data_dict
        else:
            # for soon dataset: end viewpoint is the target location <-> instructions
            item = self.data[index]
            scan = item['scan']
            viewpoint = item['path'][-1]
            view_fts, obj_img_fts, obj_attrs = self.get_scan_viewpoint_feature(scan, viewpoint)
            data_dict = {
                'question': " ".join(item['qa']['question'].split()),
                'answer': " ".join(item['qa']['answer'].split()),
                'instruction': " ".join(item['instruction'][4].split()),
                'img_feats': view_fts[:, :self.config.image_feat_size],
                'obj_feats': obj_img_fts[:, :self.config.obj_feat_size],
            }
            return data_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')

    parser.add_argument('--cfg_file', type=str, default="/home/zlin/vln/turning/VLN-DUET/qallm/cfgs/dataset.yaml", help='dataset configs')
    parser.add_argument('--img_feats', type=str, default="vit_imagenet", help='dataset configs')
    parser.add_argument('--obj_feats', type=str, default="butd_SOON", help='dataset configs')
    parser.add_argument('--split', type=str, default="train", help='dataset configs')

    args = parser.parse_args()

    dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
    dataset_cfg.Dataset.Img_Features_File_Map = dataset_cfg.Dataset.Img_Features_File_Map[args.img_feats]
    dataset_cfg.Dataset.Object_Features_File_Map = dataset_cfg.Dataset.Object_Features_File_Map[args.obj_feats]
    dataset = BaseDataset(config=dataset_cfg.Dataset,split=args.split)

    pbar = tqdm(dataset,desc="iterate dataset: ")
    for i, data_dict in enumerate(pbar):
        pass