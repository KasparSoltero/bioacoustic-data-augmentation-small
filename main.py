import json
import yaml
import os
import torchaudio
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import hsv_to_rgb
from PIL import Image
import csv
import chardet
from matplotlib.ticker import PercentFormatter

from spectrogram_tools import spectrogram_transformed, spec_to_audio, crop_overlay_waveform, load_waveform, transform_waveform, map_frequency_to_log_scale, map_frequency_to_linear_scale, merge_boxes_by_class, generate_masks, rle_decode
from old_display import plot_spectrogram
from colors import custom_color_maps, hex_to_rgb

def plot_labels(config, idx=[0,-1], save_directory='output'):
    if not config['output']['include_spectrogram']:
        print('Spectrograms are not included in the output; skipping plot_labels')
        return
    # Plotting the labelsq
    # check if species value map exists
    species_value_map = {}
    if os.path.exists(f'{save_directory}/species_value_map.csv'):
        with open(f'{save_directory}/species_value_map.csv', 'r') as f:
            for line in f:
                key, value = line.strip().split(',') #reading in reverse
                species_value_map[int(key)] =value
    # Plotting the spectrograms
    # Calculate the number of rows needed
    if idx[1] == -1:
        idx[1] = 9
    rows = (idx[1] - idx[0]) //  3 if (idx[1] - idx[0]) else 1
    if rows < 1:
        rows = 1
    if config['output']['include_masks']:
        rows *= 2
    if rows > 4:
        rows = 4
        
    # Plotting the spectrograms
    fig, axes = plt.subplots(rows, 3, figsize=(20, 4 * rows))
    fig.canvas.manager.set_window_title('') 
    fig.suptitle(f'{save_directory}/artificial_dataset/images', fontsize=12)

    # Ensure axes is always a 2D array
    axes = np.array(axes).reshape(rows, -1)

    for i, image_path in enumerate(os.listdir(f'{save_directory}/artificial_dataset/images')[idx[0]:idx[1]]):
        if image_path == '.DS_Store':
            continue
        
        # Compute row and column index
        row_idx = i // 3
        col_idx = i % 3
        if row_idx >= rows:
            break
        
        image = Image.open(f'{save_directory}/artificial_dataset/images/{image_path}')
        ax = axes[row_idx][col_idx]
        image_array = np.array(image)

        if config['output']['include_boxes']:
            label_path = f'{save_directory}/artificial_dataset/box_labels/{image_path[:-4]}.txt'
            # get the corresponding label
            boxes = []
            with open(label_path, 'r') as f:
                for line in f:
                    # Split on commas and strip whitespace
                    # values = [value.strip() for value in line.split(',')]
                    # values separated by spaces
                    values = [value.strip() for value in line.split(' ')]
                    
                    # Convert to float, ignoring empty strings
                    class_id, x_center, y_center, width, height = [float(value) for value in values if value]
                    
                    boxes.append([class_id, x_center, y_center, width, height])
    
            # plot boxes
            for box in boxes:
                x_center, y_center, width, height = box[1:]
                x_min = x_center * 10  # Multiply by 10 to match the time axis
                y_min = (1 - y_center) * 24000  # Adjust y-coordinate for upper origin
                box_width = width * 10
                box_height = height * 24000
                rect = plt.Rectangle((x_min - box_width/2, y_min - box_height/2), box_width, box_height,
                                    linewidth=1, edgecolor=custom_color_maps['teal'], facecolor='none')
                # elif box[0] == 0:
                    # rect = plt.Rectangle((x_min - box_width/2, y_min - box_height/2), box_width, box_height, 
                                        # linewidth=1, edgecolor='white', facecolor='none', linestyle='--')
                # elif box[0] == 1:
                    # rect = plt.Rectangle((x_min - box_width/2, y_min - box_height/2), box_width, box_height, 
                                        # linewidth=1, edgecolor='r', facecolor='none', linestyle='--')
                ax.add_patch(rect)
                if species_value_map and config['plot']['show_labels']:
                    labeltext = species_value_map[int(box[0])]
                    # insert newlines
                    if ' ' in labeltext:
                        labeltext = labeltext.split(' ')[0] + '\n' + labeltext.split(' ')[1]
                    ax.text(x_min + box_width/2, y_min + box_height/2, labeltext, fontsize=6, color='#eeeeee')

        if config['output']['include_masks']:
            # Load COCO annotations
            coco_path = f'{save_directory}/artificial_dataset/mask_annotations.json'
            if os.path.exists(coco_path):
                with open(coco_path, 'r') as f:
                    coco_data = json.load(f)
                
                # Find annotations for current image
                image_name = image_path[:-4]  # Remove .jpg extension
                image_id = None
                for img in coco_data['images']:
                    if img['file_name'].startswith(image_name):
                        image_id = img['id']
                        break
                
                if image_id is not None:
                    # Get all annotations for this image
                    image_annotations = [ann for ann in coco_data['annotations'] 
                                      if ann['image_id'] == image_id]
                    print(f'found {len(image_annotations)} annotations for {image_name}')
                    
                    # Create a colored mask overlay
                    mask_overlay = np.zeros_like(image_array)
                    if len(mask_overlay.shape) != 3:
                        # add 3 channels
                        mask_overlay = np.stack([mask_overlay, mask_overlay, mask_overlay], axis=-1)
                    
                    for j, ann in enumerate(image_annotations):
                        mask_counts = ann['segmentation']['counts']
                        mask_size = ann['segmentation']['size']
                        mask = rle_decode(mask_counts, mask_size)
                        
                        mask = mask.reshape(mask_size)  # Should be [freq, time]

                        # Convert mask to image size maintaining aspect ratio
                        freq_bins, time_bins = mask.shape
                        scale_freq = image_array.shape[0]/freq_bins
                        scale_time = image_array.shape[1]/time_bins
                        new_freq = int(freq_bins * scale_freq)
                        new_time = int(time_bins * scale_time)

                        mask_resized = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize(
                            (new_time, new_freq), 
                            Image.NEAREST
                        ))
                        
                        color = hex_to_rgb(custom_color_maps['rotary'][j % len(custom_color_maps['rotary'])])
                        mask_overlay[mask_resized > 0] = color
                    # invert y axis
                    mask_overlay = np.flipud(mask_overlay)

            # plot the masks on the next axes
            thisax = axes[row_idx+1][col_idx]
            thisax.imshow(mask_overlay, aspect='auto', origin='upper')
            thisax.set_xticks([])
            thisax.set_yticks([])
        #     # sum arrays
        #     # image_array = image_array + mask_overlay
        #     image_array = mask_overlay

        # Display image with mask overlay
        if config['plot']['color_filter'] == 'dusk':
            im = ax.imshow(image_array, aspect='auto', origin='upper', extent=[0, 10, 0, 24000], cmap=custom_color_maps['dusk'])
        else:
            if config['output']['rainbow_frequency']:
                im = ax.imshow(image_array, aspect='auto', origin='upper', extent=[0, 10, 0, 24000])
            else:
                im = ax.imshow(image_array, aspect='auto', origin='upper', extent=[0, 10, 0, 24000], cmap='gray')

        yticks = [0, 1000, 2000, 5000, 10000, 24000]
        logyticks = map_frequency_to_log_scale(24000, yticks)
        ax.set_yticks(logyticks)
        yticklabels = [0, 1, 2, 5, 10, 24]
        ax.set_yticklabels(yticklabels)
        ax.set_title(f'{image_path[:1]}')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Frequency (kHz)', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    plt.close()

def read_tags(path, config, default_species='unknown'):
    # reads a csv, returns dictionaries of filenames with each column's attributes
    tags_path = os.path.join(path, 'tags.csv')
    if os.path.exists(tags_path):
        with open(tags_path, 'rb') as raw_file:
            result = chardet.detect(raw_file.read())
            encoding = result['encoding']
        
        with open(tags_path, mode='r', newline='', encoding=encoding) as file:
            reader = csv.DictReader(file)
            tags_data = {}
            for row in reader:
                filename = row['filename']
                filename = filename.split('.')[:-1]
                filename = '.'.join(filename)
                tags_data[filename] = {}
                for header in reader.fieldnames:
                    tags_data[filename][header] = row[header]
                
        return tags_data
    else:
        tags_data = {}
        for f in os.listdir(path):
            for ext in config['input']['allowed_files']:
                if f.endswith(ext) and not f.startswith('.'):
                    found_filename = f.split('.')[0]
                    tags_data[found_filename] = {'filename': found_filename, 'species': default_species}
        return tags_data

def load_input_dataset(data_root, background_path, positive_path, negative_path, config):
    positive_segment_paths = []
    positive_datatags = read_tags(os.path.join(data_root, positive_path), config, 1)
    
    # Get the list of files and randomly shuffle it
    files = os.listdir(os.path.join(data_root, positive_path))
    random.shuffle(files)
    for f in files:
        if config['input']['limit_positives'] and len(positive_segment_paths) >= config['input']['limit_positives']:
            print(f'Limiting positive examples to {config["input"]["limit_positives"]}')
            break

        for ext in config['input']['allowed_files']:
            if f.endswith(ext) and not f.startswith('.'):
                found_filename = f.split('.')[:-1]
                found_filename = '.'.join(found_filename)
                #  overlay_label for tracing training data, e.g. 5th bird -> bi5
                if found_filename in positive_datatags:
                    positive_datatags[found_filename]['overlay_label'] = positive_path[:2]+str(list(positive_datatags.keys()).index(found_filename))
                else: 
                    positive_datatags[found_filename] = {'filename': found_filename, 'species': 'unknown', 'overlay_label': 'unknown'}
                full_audio_path = os.path.join(data_root, positive_path, f)
                positive_segment_paths.append(full_audio_path)
                break

    negative_segment_paths = []
    negative_datatags = read_tags(os.path.join(data_root, negative_path), config, 0)
    for f in os.listdir(os.path.join(data_root, negative_path)):
        for ext in config['input']['allowed_files']:
            if f.endswith(ext) and not f.startswith('.'):
                found_filename = f.split('.')[:-1]
                found_filename = '.'.join(found_filename)
                negative_datatags[found_filename]['overlay_label'] = negative_path[:2]+str(list(negative_datatags.keys()).index(found_filename))
                negative_segment_paths.append(os.path.join(data_root, negative_path, f))
                break

    background_noise_paths = []
    background_datatags = read_tags(os.path.join(data_root, background_path), config)
    for f in os.listdir(os.path.join(data_root, background_path)):
        for ext in config['input']['allowed_files']:
            if f.endswith(ext) and not f.startswith('.'):
                found_filename = f.split('.')[:-1]
                found_filename = '.'.join(found_filename)
                background_datatags[found_filename]['overlay_label'] = 'bg'+str(list(background_datatags.keys()).index(found_filename))
                background_noise_paths.append(os.path.join(data_root, background_path, f))
                break

    return positive_segment_paths, positive_datatags, negative_segment_paths, negative_datatags, background_noise_paths, background_datatags

def write_species_value_map_to_file(species_value_map, save_directory='output'):
    # write the species value map to a file
    with open(f'{save_directory}/species_value_map.csv', 'w') as f:
        for key, value in species_value_map.items():
            f.write(f'{value},{key}\n') # reverse order for easy reading

def find_max_index(directory):
    """Find the maximum index from existing files in a directory"""
    max_idx = -1
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            if filename.endswith('.wav'):
                try:
                    idx = int(filename.split('.')[0])
                    max_idx = max(max_idx, idx)
                except ValueError:
                    continue
    return max_idx

def generate_overlays(
        config,
        get_data_paths=[None, None, None, None],
        save_directory='datasets_mutable',
        n=1,
        sample_rate=48000,
        final_length_seconds=10,
        output_generate_masks=False,
        positive_overlay_range=[1,1],
        negative_overlay_range=[0,0],
        save_wav=False,
        plot=False,
        clear_dataset=False,
        val_ratio = 0.8,
        snr_range=[0.1,1],
        repetitions=[1,10],
        specify_positive=None,
        specify_noise='/Volumes/Rectangle/bioacoustic-data-augmentation-dataset/noise/Heavy-Rain-Falling-Off-Roof-A1-www.fesliyanstudios.com_01.wav',
        specify_bandpass=None,
        color_mode='HSV'
    ):
    # Loop for creating and overlaying spectrograms
    # DEFAULTS: 
        # noise normalised to 1 rms, dB
        # song set to localised snr 1-10
        # song bbox threshold 5 dB over 10 bands (240hz)
        # songs can be cropped over edges, minimum 1 second present
        # images are normalised to 0-100 dB, then 0-1 to 255
        # 80:20 split train and val
        # 640x640 images
    # TODO:
        # training data spacings for long ones, add distance/spacing random additions in loop

    # Determine starting index for concatenation mode
    start_idx = 0
    if config['output']['concatenate']:
        sound_files_dir = 'classifiers/augmented_dataset/sound_files'
        start_idx = find_max_index(sound_files_dir) + 1
        print(f"Concatenating from index {start_idx}")

    if clear_dataset and not config['output']['concatenate']:
        os.system(f'rm -rf {save_directory}/artificial_dataset/images/train/*')
        os.system(f'rm -rf {save_directory}/artificial_dataset/images/val/*')
        os.system(f'rm -rf {save_directory}/artificial_dataset/box_labels/train/*')
        os.system(f'rm -rf {save_directory}/artificial_dataset/box_labels/val/*')
        os.system(f'rm -rf {save_directory}/artificial_dataset/mask_annotations.json')
        os.system(f'rm -rf {save_directory}/species_value_map.csv')
        os.system(f'rm -rf classifiers/augmented_dataset/sound_files/*')
        os.system(f'rm -rf classifiers/augmented_dataset/labels.csv')

    data_root, background_path, positive_paths, negative_paths = get_data_paths
    if data_root is None:
        data_root='../data/manually_isolated'
        background_path='background_noise'
    if positive_paths is None:
        positive_paths = ['unknown', 'amphibian', 'reptile', 'mammal', 'insect', 'bird']
        negative_paths = ['anthrophony', 'geophony']
    positive_segment_paths, positive_datatags, negative_segment_paths, negative_datatags, background_noise_paths, background_datatags = load_input_dataset(data_root, background_path, positive_paths, negative_paths, config)

    # load config
    # generate species value map
    species_value_map = {}
    
    if config['output']['include_masks']:
        coco_dataset = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        coco_annotations = []
    if config['output']['include_kaytoo']:
        # Initialize the file with headers ONCE before the loop
        labels_path = f'classifiers/augmented_dataset/labels.csv'
        if not os.path.exists(labels_path):
            os.makedirs(os.path.dirname(labels_path), exist_ok=True)
            with open(labels_path, 'w') as f:
                f.write('filename,primary_label\n')

    val_index = int(n*val_ratio) # validation

    # main loop to create and overlay audio
    for idx_offset in range(n):
        idx = start_idx + idx_offset  # Use adjusted index
        if idx == n:
            break
        label = str(idx) # image label
        # Select a random background noise (keep trying until one is long enough)
        noise_db = -9
        bg_noise_waveform_cropped = None
        while bg_noise_waveform_cropped is None:
            if specify_noise is not None:
                bg_noise_path = specify_noise
            else:
                bg_noise_path = random.choice(background_noise_paths)
            bg_noise_waveform, original_sample_rate = load_waveform(bg_noise_path)
            bg_noise_waveform_cropped = transform_waveform(bg_noise_waveform, 
                resample=[original_sample_rate,sample_rate], 
                random_crop_seconds=final_length_seconds
            )
        if random.uniform(0,1)>0.5: # 50% chance add white noise 0.005 - 0.01 rms
            bg_noise_waveform_cropped = transform_waveform(bg_noise_waveform_cropped,
                add_white_noise=random.uniform(0.005, 0.03)
            )
        if random.uniform(0,1)>0.5: # 50% chance add pink noise 0.005 - 0.01 rms
            bg_noise_waveform_cropped = transform_waveform(bg_noise_waveform_cropped,
                add_pink_noise=random.uniform(0.005, 0.03)
            )
        if random.uniform(0,1)>0.5: # 50% chance add brown noise 0.005 - 0.01 rms
            bg_noise_waveform_cropped = transform_waveform(bg_noise_waveform_cropped,
                add_brown_noise=random.uniform(0.005, 0.03)
            )
        # bg_noise_waveform_cropped = transform_waveform(bg_noise_waveform_cropped,
        #     add_pink_noise=0.005
        # )
        # set db
        bg_noise_waveform_cropped = transform_waveform(bg_noise_waveform_cropped, set_db=noise_db)

        label += '_' + background_datatags[os.path.basename(bg_noise_path)[:-4]]['overlay_label']

        # highpass filter set by background noise tags data
        highpass_hz = background_datatags[os.path.basename(bg_noise_path)[:-4]].get('highpass', None)
        if highpass_hz:
            highpass_hz = int(highpass_hz)
        else:
            highpass_hz = 0
        highpass_hz += random.randint(0,config['output']['highpass_variable'])
        lowpass_hz = background_datatags[os.path.basename(bg_noise_path)[:-4]].get('lowpass', None)
        if lowpass_hz:
            lowpass_hz = int(lowpass_hz)
        else:
            lowpass_hz = (min(original_sample_rate,sample_rate)) / 2
        lowpass_hz -= random.randint(0,config['output']['lowpass_variable'])
        if specify_bandpass is not None:
            highpass_hz, lowpass_hz = specify_bandpass

        # adding random number of negative noises (cars, rain, wind). 
        # no boxes stored for these, as they are treated like background noise
        n_negative_overlays = random.randint(negative_overlay_range[0], negative_overlay_range[1])
        for j in range(n_negative_overlays):
            negative_segment_path = random.choice(negative_segment_paths)
            label += '_'+negative_datatags[os.path.basename(negative_segment_path)[:-4]]['overlay_label']
            
            negative_waveform, neg_sr = load_waveform(negative_segment_path)

            neg_db = 10*torch.log10(torch.tensor(random.uniform(snr_range[0], snr_range[1])))+noise_db
            negative_waveform = transform_waveform(negative_waveform, resample=[neg_sr,sample_rate], set_db=neg_db)
            
            negative_waveform_cropped, start = crop_overlay_waveform(bg_noise_waveform_cropped.shape[1], negative_waveform)

            overlay = torch.zeros_like(bg_noise_waveform_cropped)
            overlay[:,max(0,start) : max(0,start) + negative_waveform_cropped.shape[1]] = negative_waveform_cropped
            bg_noise_waveform_cropped += overlay

            label += 'p' + f"{(10 ** ((neg_db - noise_db) / 10)).item():.3f}" # power label
            
        new_waveform = bg_noise_waveform_cropped.clone()
        bg_spec_temp = transform_waveform(bg_noise_waveform_cropped, to_spec='power')
        bg_time_bins, bg_freq_bins = bg_spec_temp.shape[2], bg_spec_temp.shape[1]
        freq_bins_cutoff_bottom = int((highpass_hz / (sample_rate / 2)) * bg_freq_bins)
        freq_bins_cutoff_top = int((lowpass_hz / (sample_rate / 2)) * bg_freq_bins)

        # Adding random number of positive vocalisation noises
        # initialise label arrays
        boxes = []
        classes = []
        n_positive_overlays = random.randint(positive_overlay_range[0], positive_overlay_range[1])
        print(f'\n{idx}:    creating new image with {n_positive_overlays} positive overlays, bg={os.path.basename(bg_noise_path)}')
        succuessful_positive_overlays = 0
        while_catch = 0
        while succuessful_positive_overlays < n_positive_overlays:
            while_catch += 1
            if while_catch > 100:
                print(f"{idx}: Error, too many iterations")
                break

            # select positive overlay
            if specify_positive is not None:
                positive_segment_path = specify_positive
            else: 
                positive_segment_path = random.choice(positive_segment_paths)
                    
            # check if 'species' is 'chorus' (regardless of single_class because this determines how things are placed in the 10s image)
            # if positive_datatags[os.path.basename(positive_segment_path)[:-4]]['species'] == 'chorus':
                # continue # #TODO fix skip chorus
                # if classes.count(1) > 0:
                    # continue # only one chorus per image
                # species_class=1
            # else:
                # species_class=0
            species_class = positive_datatags[os.path.basename(positive_segment_path)[:-4]].get('species', None)
            if not species_class:
                species_class = 'unknown'

            positive_waveform, pos_sr = load_waveform(positive_segment_path)
            positive_waveform = transform_waveform(positive_waveform, resample=[pos_sr,sample_rate])
            positive_waveform_cropped, start = crop_overlay_waveform(bg_noise_waveform_cropped.shape[1], positive_waveform)
            
            # attempt to place segment at least 1 seconds from other starts #TODO this introduces a bias
            if positive_waveform.shape[1] < bg_noise_waveform_cropped.shape[1]:
                for i in range(20):
                    positive_waveform_cropped, start = crop_overlay_waveform(bg_noise_waveform_cropped.shape[1], positive_waveform)
                    if not any([start < box[0] + 1*sample_rate and start > box[0] - 1*sample_rate for box in boxes]):
                        break

            threshold = 2 # PSNR, db
            band_check_width = 5 # 5 bins
            edge_avoidance = 0.005 # 0.5% of final image per side, 50 milliseconds 120 Hz rounds to 4 and 5 bins -> 43 milliseconds 117 Hz
            freq_edge, time_edge = int(edge_avoidance*bg_freq_bins), int(edge_avoidance*bg_time_bins)
            # first pass find frequency top and bottom
            positive_spec_temp = transform_waveform(positive_waveform_cropped, to_spec='power')
            seg_freq_bins, seg_time_bins = positive_spec_temp.shape[1], positive_spec_temp.shape[2]
            start_time_bins = int(start * bg_time_bins / bg_noise_waveform_cropped.shape[1])
            first_pass_freq_start, first_pass_freq_end=None, None
            for i in range(max(freq_edge,freq_bins_cutoff_bottom), min(seg_freq_bins-freq_edge,freq_bins_cutoff_top)-1-band_check_width):
                PS_avg = torch.mean(torch.tensor([positive_spec_temp[:,j:j+1,:].max() for j in range(i,i+band_check_width)]))
                N_avg = torch.mean(torch.tensor([
                    bg_spec_temp[:,
                        j:j+1,
                        max(start_time_bins,time_edge):min(start_time_bins+seg_time_bins,bg_time_bins-time_edge)
                    ].mean() for j in range(i,i+band_check_width)]
                ))
                if (10*torch.log10(PS_avg / N_avg) > threshold) and (PS_avg > threshold):
                    first_pass_freq_start = i
                    break
            for i in range(min(seg_freq_bins-freq_edge, freq_bins_cutoff_top)-1, max(freq_edge,freq_bins_cutoff_bottom)+band_check_width, -1):
                PS_avg = torch.mean(torch.tensor([positive_spec_temp[:,j:j+1,:].max() for j in range(i-band_check_width,i)]))
                N_avg = torch.mean(torch.tensor([
                    bg_spec_temp[:,
                        j:j+1,
                        max(start_time_bins,time_edge):min(start_time_bins+seg_time_bins,bg_time_bins-time_edge)
                    ].mean() for j in range(i-band_check_width,i)]
                ))
                if (10*torch.log10(PS_avg / N_avg) > threshold) and (PS_avg > threshold):
                    first_pass_freq_end = i
                    break
            if (first_pass_freq_start and first_pass_freq_end) and (first_pass_freq_end > first_pass_freq_start) and (start_time_bins+seg_time_bins < bg_time_bins):
                #calculate noise power at box
                full_spec = torch.zeros_like(bg_spec_temp[:, :, max(0,start_time_bins):start_time_bins+seg_time_bins])
                full_spec[:, first_pass_freq_start:first_pass_freq_end, :] = bg_spec_temp[:, first_pass_freq_start:first_pass_freq_end, max(0,start_time_bins):start_time_bins+seg_time_bins]
                waveform_at_box = torchaudio.transforms.GriffinLim(
                    n_fft=2048, 
                    win_length=2048, 
                    hop_length=512, 
                    power=2.0
                )(full_spec)
                noise_db_at_box = 10*torch.log10(torch.mean(torch.square(waveform_at_box)))

                pos_snr = torch.tensor(random.uniform(snr_range[0], snr_range[1]))
                pos_db = 10*torch.log10(pos_snr)+noise_db_at_box
                # power shift signal
                positive_waveform_cropped = transform_waveform(positive_waveform_cropped, set_db=pos_db)
                # dynamically find the new bounding box after power shift
                pos_spec_temp = transform_waveform(positive_waveform_cropped, to_spec='power')
            else:
                continue

            found=0
            # if seg_time_bins < bg_time_bins:
            if True:
                # Find frequency edges (vertical scan)
                freq_start = max(freq_edge,freq_bins_cutoff_bottom) # from the bottom up
                for i in range(max(freq_edge,freq_bins_cutoff_bottom), min(seg_freq_bins-freq_edge,freq_bins_cutoff_top)-1-band_check_width):
                    N_avg = torch.mean(torch.tensor([
                        bg_spec_temp[:,
                            j:j+1,
                            max(start_time_bins,time_edge):min(start_time_bins+seg_time_bins,bg_time_bins-time_edge)
                        ].mean() for j in range(i,i+band_check_width)]
                    ))
                    PS_avg = torch.mean(torch.tensor([pos_spec_temp[:,j:j+1,:].max() for j in range(i,i+band_check_width)]))
                    if (10*torch.log10(PS_avg / N_avg) > threshold) and (PS_avg > threshold):
                        freq_start = i
                        found+=1
                        break

                freq_end = min(seg_freq_bins-freq_edge, freq_bins_cutoff_top)-1 # from the top down
                for i in range(min(seg_freq_bins-freq_edge, freq_bins_cutoff_top)-1, max(freq_edge,freq_bins_cutoff_bottom)+band_check_width, -1):
                    N_avg = torch.mean(torch.tensor([
                        bg_spec_temp[:,
                            j:j+1,
                            max(start_time_bins,time_edge):min(start_time_bins+seg_time_bins,bg_time_bins-time_edge)
                        ].mean() for j in range(i-band_check_width,i)]
                    ))
                    PS_avg = torch.mean(torch.tensor([pos_spec_temp[:,j:j+1,:].max() for j in range(i-band_check_width,i)]))
                    if (10*torch.log10(PS_avg / N_avg) > threshold) and (PS_avg > threshold):
                        freq_end = i
                        found+=1
                        break

                # Find time edges (horizontal scan)
                start_time_offset = 0 # from the left
                if freq_start < freq_end:
                    for i in range(0, seg_time_bins-1-band_check_width):
                        N_avg = torch.mean(torch.tensor([
                            bg_spec_temp[:,
                                freq_start:freq_end,
                                j:j+1
                            ].mean() for j in range(i,i+band_check_width)]
                        ))
                        PS_avg = torch.mean(torch.tensor([pos_spec_temp[:,freq_start:freq_end,j:j+1].max() for j in range(i,i+band_check_width)]))
                        if (10*torch.log10(PS_avg / N_avg) > threshold) and (PS_avg > threshold):
                            start_time_offset = i
                            found+=1
                            break

                    end_time_offset = seg_time_bins - 1 # from the right
                    for i in range(seg_time_bins - 1, 0+band_check_width, -1):
                        N_avg = torch.mean(torch.tensor([
                            bg_spec_temp[:,
                                freq_start:freq_end,
                                j:j+1
                            ].mean() for j in range(i-band_check_width,i)]
                        ))
                        PS_avg = torch.mean(torch.tensor([pos_spec_temp[:,freq_start:freq_end,j:j+1].max() for j in range(i-band_check_width,i)]))
                        if (10*torch.log10(PS_avg / N_avg) > threshold) and (PS_avg > threshold):
                            end_time_offset = i
                            found+=1
                            break

            # TODO maybe remove?: noises longer than final length are treated as continuous, no need for time edges
            #TODO: tripple check iou ios merging calcualtions due to format change
            elif seg_time_bins >= bg_time_bins:
                # Find frequency edges (vertical scan) - minimum start at 2 (~100 Hz @ 48khz) to avoid low frequency interferance
                freq_start = freq_edge
                for i in range(max(freq_edge,freq_bins_cutoff_bottom), min(seg_freq_bins-freq_edge,freq_bins_cutoff_top)-1):
                    N = bg_spec_temp[:,
                        i:i+1,time_edge:bg_time_bins-time_edge
                    ].mean()
                    PS = pos_spec_temp[:,i:i+1,time_edge:seg_time_bins-time_edge].max()
                    if (10*torch.log10(PS / N) > threshold) and (PS > threshold):
                        freq_start = i
                        found+=1
                        break
                freq_end = seg_freq_bins - 1
                for i in range(min(seg_freq_bins, freq_bins_cutoff_top)-1, max(2,freq_bins_cutoff_bottom), -1):
                    N = bg_spec_temp[:,
                        i:i+1,time_edge:bg_time_bins-time_edge
                    ].mean()
                    PS = pos_spec_temp[:,i:i+1,time_edge:seg_time_bins-time_edge].max()
                    if (10*torch.log10(PS / N) > threshold) and (PS > threshold):
                        freq_end = i
                        found+=1
                        break
                if freq_start < freq_end:
                    start_time_offset = 0
                    end_time_offset = seg_time_bins - 1
                    found+=2

            # verify height and width are not less than 1% of the final image
            if ((freq_end - freq_start)/bg_freq_bins) < 0.0065 or ((end_time_offset - start_time_offset)/bg_time_bins) < 0.0065:
                print(f"{idx}: Error, too small, power {pos_db-noise_db:.3f}, freq {(freq_end - freq_start)/bg_freq_bins:.3f}, time {(end_time_offset - start_time_offset)/bg_time_bins:.3f}")
                continue
            if ((freq_end - freq_start)/bg_freq_bins) > 0.99 or found < 4:
                print(f"{idx}: Error, too faint, power {pos_db-noise_db:.3f}")
                continue

            ## Paper small square Plot
            # combined_for_plot = bg_noise_waveform_cropped.clone()
            # combined_for_plot[:,max(0,start) : max(0,start) + positive_waveform_cropped.shape[1]] += positive_waveform_cropped
            # temp_comobined_spec = transform_waveform(combined_for_plot, to_spec='power')
            # plot_spectrogram(paths=['x'], not_paths_specs=[temp_comobined_spec],
            #     logscale=False, 
            #     color='bw',
            #     draw_boxes=[[
            #         [10, seg_time_bins+10, first_pass_freq_start, first_pass_freq_end],
            #         [start_time_offset+10, end_time_offset+10, freq_start, freq_end]
            #         ]],
            #     box_format='xxyy',
            #     set_width=1,fontsize=15,
            #     box_colors=['#00eaff','#45ff45'],
            #     box_styles=['solid','--'],
            #     box_widths=[3,3],
            #     crop_time=[max(0,start_time_bins-15), min(start_time_bins+seg_time_bins+15,bg_time_bins)],
            #     crop_frequency=[max(first_pass_freq_start-15,0), min(first_pass_freq_end+15,bg_freq_bins)],
            #     specify_freq_range=[((first_pass_freq_start-15)/bg_freq_bins)*24000, ((first_pass_freq_end+15)/bg_freq_bins)*24000]
            # )

            def appendSpeciesClass(classes, species_class):
                # print(f' {species_class} ', end='')
                if species_class in species_value_map:
                    classes.append(species_value_map[species_class])
                else:
                    classes.append(len(species_value_map))
                    species_value_map[species_class] = len(species_value_map)
                    write_species_value_map_to_file(species_value_map, save_directory)
                # print(f'    {species_class}')
                return classes

            overlay = torch.zeros_like(bg_noise_waveform_cropped)
            overlay[:,max(0,start) : max(0,start) + positive_waveform_cropped.shape[1]] = positive_waveform_cropped
            new_waveform += overlay
            succuessful_positive_overlays += 1

            freq_start, freq_end = map_frequency_to_log_scale(bg_freq_bins, [freq_start, freq_end])
            # add bounding box to list, in units of spectrogram time and log frequency bins
            boxes.append([max(start_time_offset,start_time_bins+start_time_offset), max(end_time_offset, start_time_bins+end_time_offset), freq_start, freq_end])
            classes = appendSpeciesClass(classes, species_class)
            label += positive_datatags[os.path.basename(positive_segment_path)[:-4]]['overlay_label']
            label += 'p' + f"{pos_snr:.1f}" # power label

            if output_generate_masks:
                # Generate mask annotation
                mask_annotation = generate_masks(
                    overlay_waveform=overlay,
                    image_id=idx,
                    category_id=classes[-1],
                    last_box=boxes[-1],
                    threshold_db=10,
                    log_scale=True,
                    debug=False
                )
                coco_annotations.append(mask_annotation)

            # potentially repeat song
            if repetitions:
                if random.uniform(0,1)>0.5:
                    seg_samples = positive_waveform_cropped.shape[1]
                    separation = random.uniform(0.5, 2) # 0.5-3 seconds
                    separation_samples = int(separation*sample_rate)
                    n_repetitions = random.randint(repetitions[0], repetitions[1])
                    print(f'{idx}:    repeating {n_repetitions} times, separation {separation:.2f}s')
                    new_start = start
                    for i in range(n_repetitions):
                        new_start += seg_samples + separation_samples
                        if new_start + seg_samples < (bg_noise_waveform_cropped.shape[1]-1) and (new_start>0):
                            new_start_bins = int(new_start * bg_time_bins / bg_noise_waveform_cropped.shape[1])
                            overlay = torch.zeros_like(bg_noise_waveform_cropped)
                            overlay[:,new_start : new_start + positive_waveform_cropped.shape[1]] = positive_waveform_cropped
                            new_waveform += overlay
                            succuessful_positive_overlays += 1

                            boxes.append([new_start_bins+start_time_offset, new_start_bins+end_time_offset, freq_start, freq_end])
                            classes = appendSpeciesClass(classes, species_class)
                            label += 'x' # repetition
                            if output_generate_masks:
                                mask_annotation = generate_masks(
                                    overlay_waveform=overlay,
                                    image_id=idx,
                                    category_id=classes[-1],
                                    last_box=boxes[-1],
                                    threshold_db=10,
                                    log_scale=True,
                                    debug=False
                                )
                                coco_annotations.append(mask_annotation)
                        else:
                            break
            
        final_audio = transform_waveform(new_waveform, to_spec='power')
        final_audio = spectrogram_transformed(
            final_audio,
            highpass_hz=highpass_hz,
            lowpass_hz=lowpass_hz
        )

        # final normalisation, which is applied to real audio also
        final_audio = spectrogram_transformed(
            final_audio,
            set_db=-10,
        )
        if not config['paths']['do_train_val_split']:
            save_files_path = f"{idx}"
        elif idx_offset > val_index:
            save_files_path = f"val/{idx}"
        else:
            save_files_path = f"train/{idx}"

        if config['output']['include_kaytoo']:
            # Append to the file within the loop
            wav_path = f"classifiers/augmented_dataset/sound_files/{save_files_path}"
            if not os.path.exists(os.path.dirname(wav_path)):
                os.makedirs(os.path.dirname(wav_path))
            spec_to_audio(final_audio, save_to=wav_path, energy_type='power')
            
            coarse_labels_output_path = f'classifiers/augmented_dataset/labels.csv'
            with open(coarse_labels_output_path, 'a') as f:
                if config['output']['single_class']:
                    if len(classes) > 0:
                        coarse_class = 1
                    else:
                        coarse_class = 0
                f.write(f'{idx}.wav,{coarse_class}\n')
        
        if config['output']['include_spectrogram']:
            image = spectrogram_transformed(
                final_audio,
                to_pil=True,
                color_mode=color_mode,
                log_scale=True,
                normalise='power_to_PCEN',
                resize=(640, 640),
            )
            image_output_path = f'{save_directory}/artificial_dataset/images/{save_files_path}.jpg'
        
            # check directory exists
            if not os.path.exists(os.path.dirname(image_output_path)):
                os.makedirs(os.path.dirname(image_output_path))
            image.save(image_output_path, format='JPEG', quality=95)
            # Reopen the image to check for errors (slow)
            # try:
            #     img = Image.open(image_output_path)
            #     img.load()  # loading of image data
            #     img.close()
            # except (IOError, SyntaxError) as e:
            #     print(f"Invalid image after reopening: {e}")

        if config['output']['include_boxes']:
            box_label_output_path = f'{save_directory}/artificial_dataset/box_labels/{save_files_path}.txt'

            # Merge boxes based on IoU
            merged_boxes, merged_classes = merge_boxes_by_class(boxes, classes, iou_threshold=0.1, ios_threshold=0.4)
            
            # use this to remember how to turn off log later
            # temp_unlog_boxes = []
            # for box in boxes:
            #     y1, y2 = map_frequency_to_linear_scale(bg_freq_bins, [box[2], box[3]])
            #     temp_unlog_boxes.append([box[0], box[1], y1, y2])
            # plot_spectrogram(
            #     paths=['x'],
            #     not_paths_specs=[final_audio],
            #     logscale=True,fontsize=16,set_width=1.5,
            #     draw_boxes=[temp_unlog_boxes],
            #     box_colors=['#45ff45']*len(boxes),
            #     box_widths=[2]*len(boxes),
            #     box_format='xxyy')
            # temp_unlog_boxes = []
            # for box in merged_boxes:
            #     y1, y2 = map_frequency_to_linear_scale(bg_freq_bins, [box[2], box[3]])
            #     temp_unlog_boxes.append([box[0], box[1], y1, y2])
            # temp_pcen_spec = pcen(final_audio)
            # plot_spectrogram(
            #     paths=['x'],
            #     not_paths_specs=[temp_pcen_spec],color_mode='HSV',to_db=False,
            #     logscale=True,fontsize=15,set_width=1.3,
            #     draw_boxes=[temp_unlog_boxes],
            #     box_colors=['white']*len(merged_boxes),
            #     box_widths=[2]*len(merged_boxes),
            #     box_format='xxyy')
            
            # make label txt file
            # check directory exists
            if not os.path.exists(os.path.dirname(box_label_output_path)):
                os.makedirs(os.path.dirname(box_label_output_path))
            with open(box_label_output_path, 'w') as f:
                for box, species_class in zip(merged_boxes, merged_classes):
                    x_center = (box[0] + box[1]) / 2 / bg_time_bins
                    width = (box[1] - box[0]) / bg_time_bins

                    y_center = (box[2] + box[3]) / 2 / bg_freq_bins
                    y_center = 1 - y_center # vertical flipping for yolo
                    height = (box[3] - box[2]) / bg_freq_bins

                    if x_center < 0 or x_center > 1 or y_center < 0 or y_center > 1 or width < 0 or width > 1 or height < 0 or height > 1:
                        print(f"{idx}: Error, box out of bounds!\n\n******\n\n******\n\n*******\n\n")

                    # Write to file in the format [class_id x_center y_center width height]
                    f.write(f'{species_class} {x_center} {y_center} {width} {height}\n')
        
        if config['output']['include_masks']:
            # Add images info
            for i in range(n):
                coco_dataset['images'].append({
                    'id': i,
                    'file_name': f'{i}.jpg',
                    'width': 640,
                    'height': 640
                })
            
            # Add categories
            for species, idx in species_value_map.items():
                coco_dataset['categories'].append({
                    'id': idx,
                    'name': species,
                    'supercategory': 'vocalisation'
                })
            
            # Add annotations
            coco_dataset['annotations'] = coco_annotations
            
            # Save COCO dataset
            with open(f'{save_directory}/artificial_dataset/mask_annotations.json', 'w') as f:
                json.dump(coco_dataset, f)

    if(plot):
        plot_labels(config, [0,n], save_directory)

def run_augmentation(config=None):
    """Main function to run the augmentation pipeline with given config"""
    if config is None:
        # Load default config if none provided
        with open('config.yaml') as f:
            config = yaml.safe_load(f)
    
    dataset_path = config['paths']['dataset']
    background_path = config['paths']['noise']
    positive_paths = config['paths']['vocalisations']
    negative_paths = config['paths']['negative']

    if config['output']['rainbow_frequency']:
        color_mode = 'HSV'
    else:
        color_mode = 'BW'

    print(f'Generating overlays for {config["output"]["n"]} images')

    # generate overlays
    generate_overlays(
        config,
        get_data_paths = [dataset_path, background_path, positive_paths, negative_paths],
        save_directory = config['paths']['output'],
        n=config['output']['n'],
        clear_dataset=config['output']['overwrite_output_path'],
        sample_rate=48000,
        final_length_seconds=config['output']['length'],
        output_generate_masks=config['output']['include_masks'],
        positive_overlay_range=config['output']['positive_overlay_range'],
        negative_overlay_range=config['output']['negative_overlay_range'],
        val_ratio=config['output']['val_ratio'],
        snr_range=config['output']['snr_range'],
        save_wav=config['output']['include_soundfile'],
        plot=config['plot']['toggle'],
        color_mode=color_mode,
        repetitions=config['output']['repetitions'],
    )

if __name__ == "__main__":
    run_augmentation()