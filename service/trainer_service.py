import json
import os
from collections import namedtuple

Sample = namedtuple('Sample', ('path', 'label', 'label_id'))


class SampleService:
    SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../images')

    @staticmethod
    def fetch_samples() -> [Sample]:
        samples = []
        labels = {}
        current_id = 1
        for root, dirs, files in os.walk(SampleService.SAMPLES_DIR):
            dir_name = os.path.basename(root).replace(' ', '-').lower()
            img_files = filter(lambda filename: filename.endswith('png') or filename.endswith('jpg'), files)
            for file in img_files:
                if dir_name not in labels:
                    labels[dir_name] = current_id
                    current_id += 1
                samples.append(Sample(path=os.path.join(root, file), label=dir_name, label_id=labels[dir_name]))
        return samples

    @staticmethod
    def create_id_label_map(samples: [Sample]):
        id_label_map = {sample.label_id: sample.label for sample in samples}

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../labels.json'), 'w') as f:
            f.write(json.dumps(id_label_map))

        return id_label_map
