import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing
import functools

sys.path.append('..')
from utilities import processArguments, sort_key


def getBoxStatistics(seq_id, seq_paths, csv_paths, n_seq, img_ext='jpg'):
    seq_path, csv_path = seq_paths[seq_id], csv_paths[seq_id]

    src_files = [os.path.join(seq_path, k) for k in os.listdir(seq_path) if
                 os.path.splitext(k.lower())[1][1:] == img_ext]
    src_files.sort(key=sort_key)

    n_frames = len(src_files)

    print('seq {}/ {} : reading data for {:d} images from {:s}...'.format(seq_id+1, n_seq, n_frames, csv_path))

    df = pd.read_csv(csv_path)

    _size = []
    _size_norm = []
    _location = []
    _location_norm = []
    _aspect_ratio = []

    for frame_id in range(n_frames):

        file_path = src_files[frame_id]
        filename = os.path.basename(file_path)

        multiple_instance = df.loc[df['filename'] == filename]
        # Total # of object instances in a file
        n_bboxes = len(multiple_instance.index)
        # Remove from df (avoids duplication)
        df = df.drop(multiple_instance.index[:n_bboxes])

        frame_data = []

        generic_target_id = -1

        for box_id in range(n_bboxes):

            bbox = multiple_instance.iloc[box_id]
            try:
                target_id = bbox['target_id']
            except KeyError:
                target_id = generic_target_id
                generic_target_id -= 1

            xmin = bbox.loc['xmin']
            ymin = bbox.loc['ymin']
            xmax = bbox.loc['xmax']
            ymax = bbox.loc['ymax']

            img_h = bbox.loc['height']
            img_w = bbox.loc['width']

            box_h, box_w = ymax - ymin, xmax - xmin
            box_cy, box_cx = (ymax + ymin) / 2.0, (xmax + xmin) / 2.0

            box_ar = float(box_w) / float(box_h)
            box_h_n, box_w_n = float(box_h) / float(img_h), float(box_w) / float(img_w)
            box_cy_n, box_cx_n = float(box_cy) / float(img_h), float(box_cx) / float(img_w)

            _size.append((box_w, box_h))
            _size_norm.append((box_w_n, box_h_n))
            _location.append((box_cx, box_cy))
            _location_norm.append((box_cx_n, box_cy_n))
            _aspect_ratio.append(box_ar)

    return _size, _size_norm, _location, _location_norm, _aspect_ratio, n_frames


def main():
    params = {
        'seq_paths': ['detrac_1_MVI_20011', 'detrac_5_MVI_20034'],
        'seq_prefix': '',
        'root_dirs': ['/data/DETRAC/Images', ],
        'save_file_name': '',
        'csv_paths': '',
        'csv_root_dir': '',
        'map_folder': '',
        'load_path': '',
        'out_dir': '',
        'n_classes': 4,
        'data_type': 'annotations',
        'img_ext': 'png',
        'batch_size': 1,
        'show_img': 0,
        'save_video': 1,
        'n_bins': 100,
        'codec': 'H264',
        'fps': 20,
        'load_results': 1,
    }

    _args = [k for k in sys.argv[1:] if not k.startswith('visualizer.')]
    vis_args = ['--{}'.format(k.replace('visualizer.', '')) for k in sys.argv[1:] if k.startswith('visualizer.')]

    processArguments(_args, params)
    seq_paths = params['seq_paths']
    root_dirs = params['root_dirs']
    csv_paths = params['csv_paths']
    csv_root_dir = params['csv_root_dir']
    data_type = params['data_type']
    n_bins = params['n_bins']
    seq_prefix = params['seq_prefix']
    out_dir = params['out_dir']
    load_results = params['load_results']

    if seq_paths and len(seq_paths) ==1 and not seq_paths[0]:
        seq_paths = []

    if root_dirs and len(root_dirs) > 1:
        n_root_dirs = len(root_dirs)
        if seq_paths:
            if len(seq_paths) != n_root_dirs:
                raise AssertionError('No. of root_dirs and seq_paths must be identical for root_dirs > 1')

            for i, root_dir in enumerate(root_dirs):
                seq_path = seq_paths[i]
                if os.path.isfile(seq_path):
                    seq_path = [x.strip() for x in open(seq_path).readlines() if x.strip()]
                else:
                    seq_path = seq_path.split(',')
                if root_dirs:
                    seq_path = [os.path.join(root_dirs, name) for name in seq_path]
        else:
            seq_paths = []
            for root_dir in root_dirs:
                seq_paths += [os.path.join(root_dir, name) for name in os.listdir(root_dir) if
                              os.path.isdir(os.path.join(root_dir, name))]
            seq_paths.sort(key=sort_key)
    else:
        if seq_paths:
            all_seq_paths = []
            for _seq_paths in seq_paths:
                if os.path.isfile(_seq_paths):
                    all_seq_paths += [x.strip() for x in open(_seq_paths).readlines() if x.strip()]
                else:
                    all_seq_paths += _seq_paths.split(',')
            if root_dirs:
                all_seq_paths = [os.path.join(root_dirs[0], name) for name in all_seq_paths]
            seq_paths = all_seq_paths

        elif root_dirs:
            root_dirs = root_dirs[0]
            seq_paths = [os.path.join(root_dirs, name) for name in os.listdir(root_dirs) if
                         os.path.isdir(os.path.join(root_dirs, name))]
            seq_paths.sort(key=sort_key)
        else:
            raise IOError('Either seq_paths or root_dir must be provided')

    if csv_paths:
        if os.path.isfile(csv_paths):
            csv_paths = [x.strip() for x in open(csv_paths).readlines() if x.strip()]
        else:
            csv_paths = csv_paths.split(',')
        if csv_root_dir:
            csv_paths = [os.path.join(csv_root_dir, name) for name in csv_paths]
    elif csv_root_dir:
        csv_paths = [os.path.join(csv_root_dir, name) for name in os.listdir(csv_root_dir) if
                     os.path.isfile(os.path.join(csv_root_dir, name)) and name.endswith('.csv')]
        csv_paths.sort(key=sort_key)
    else:
        csv_paths = [os.path.join(seq_path, data_type + '.csv') for seq_path in seq_paths]

    seq_path_ids = []

    if seq_prefix:
        seq_path_ids = [_id for _id, seq_path in enumerate(seq_paths) if
                        os.path.basename(seq_path).startswith(seq_prefix)]
        seq_paths = [seq_paths[_id] for _id in seq_path_ids]
        csv_paths = [csv_paths[_id] for _id in seq_path_ids]

    n_seq, n_csv = len(seq_paths), len(csv_paths)
    if n_seq != n_csv:
        raise IOError('Mismatch between image {} and annotation {} lengths'.format(n_seq, n_csv))

    if not out_dir:
        out_dir = '../log/box_statistics'

    os.makedirs(out_dir, exist_ok=True)

    raw_data_out_fname = os.path.join(out_dir, 'raw_data.npz')
    hist_data_out_fname = os.path.join(out_dir, 'hist_data.npz')

    if load_results:
        print('Loading results from: {}'.format(out_dir))
        raw_data = np.load(raw_data_out_fname, allow_pickle=True)
        hist_data = np.load(hist_data_out_fname, allow_pickle=True)

        box_sizes = raw_data['box_sizes']
        box_sizes_norm = raw_data['box_sizes_norm']
        box_locations = raw_data['box_locations']
        box_locations_norm = raw_data['box_locations_norm']
        box_aspect_ratios = raw_data['box_aspect_ratios']


        _box_sizes_norm = hist_data['box_sizes_norm']
        _box_locations_norm = hist_data['box_locations_norm']

        _box_aspect_ratios = hist_data['box_aspect_ratios']

        # bins = np.add.reduceat(_box_aspect_ratios[1], range(0, n_bins+1, 2))
        # _box_aspect_ratios = np.vstack((_box_aspect_ratios[0], _box_aspect_ratios[1][:-1]))

        return

    print('Saving results to: {}'.format(out_dir))

    # box_sizes = []
    # box_sizes_norm = []
    # box_locations = []
    # box_locations_norm = []
    # box_aspect_ratios = []

    n_cpus = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(n_cpus)
    print('Loading csv detections using {} threads'.format(n_cpus))
    _start_t = time.time()
    box_data = pool.map(functools.partial(
        getBoxStatistics,
        seq_paths=seq_paths,
        csv_paths=csv_paths,
        n_seq=n_seq,
    ), range(n_seq))
    _end_t = time.time()

    total_frames = np.sum([box_data[i][5] for i in range(n_seq)])
    _fps = total_frames / float(_end_t - _start_t)
    print('Done reading data for {} frames at {:.4f} fps'.format(total_frames, _fps))


    box_sizes = np.asarray([item for sublist in box_data for item in sublist[0]])
    box_sizes_norm = np.asarray([item for sublist in box_data for item in sublist[1]])
    box_locations = np.asarray([item for sublist in box_data for item in sublist[2]])
    box_locations_norm = np.asarray([item for sublist in box_data for item in sublist[3]])
    box_aspect_ratios = np.asarray([item for sublist in box_data for item in sublist[4]])

    np.savez_compressed(raw_data_out_fname,
                        box_sizes=box_sizes,
                        box_sizes_norm=box_sizes_norm,
                        box_locations=box_locations,
                        box_locations_norm=box_locations_norm,
                        box_aspect_ratios=box_aspect_ratios,
                        )

    # for i in range(n_seq):
    #     print('Processing sequence {}/{} : {}'.format(i+1, n_seq, seq_paths[i]))
    #     _size, _size_norm, _location, _location_norm, _aspect_ratio = getBoxStatistics(
    #         seq_paths[i], csv_paths[i])
    #
    #     box_sizes += _size
    #     box_sizes_norm += _size_norm
    #     box_locations += _location
    #     box_locations_norm += _location_norm
    #     box_aspect_ratios += _aspect_ratio

    box_sizes = np.asarray(box_sizes)
    box_sizes_norm = np.asarray(box_sizes_norm)
    box_sizes_norm = np.asarray(box_sizes_norm)
    box_locations_norm = np.asarray(box_locations_norm)
    box_aspect_ratios = np.asarray(box_aspect_ratios)

    fig = plt.figure()
    _box_aspect_ratios = plt.hist(box_aspect_ratios, n_bins, density=True, facecolor='g', alpha=0.75)
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Histogram of box_aspect_ratios')
    fig.tight_layout()
    out_fname = os.path.join(out_dir, 'box_aspect_ratios.png')
    fig.savefig(out_fname, dpi=300)
    # plt.axis([0, 2, 0, 2])

    fig = plt.figure()
    _box_sizes_norm = plt.hist2d(box_sizes_norm[:, 0], box_sizes_norm[:, 1], n_bins, density=True)
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Histogram of box_sizes_norm')
    fig.tight_layout()
    out_fname = os.path.join(out_dir, 'box_sizes_norm.png')
    fig.savefig(out_fname, dpi=300)
    # plt.axis([0, 0.10, 0, 0.20])

    fig = plt.figure()
    _box_locations_norm = plt.hist2d(box_locations_norm[:, 0], box_locations_norm[:, 1], n_bins, density=True)
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Histogram of box_locations_norm')
    fig.tight_layout()
    out_fname = os.path.join(out_dir, 'box_locations_norm.png')
    fig.savefig(out_fname, dpi=300)
    # plt.axis([0.4, 1, 0, 0.2])

    np.savez_compressed(hist_data_out_fname,
                        box_sizes_norm=_box_sizes_norm,
                        box_locations_norm=_box_locations_norm,
                        box_aspect_ratios=_box_aspect_ratios,
                        )
    # plt.show()

    print()


if __name__ == '__main__':
    _quit = 0
    _pause = 1

    main()
