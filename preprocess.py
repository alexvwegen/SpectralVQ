import os
import numpy as np
import hashlib
import json
import argparse

from audio_pipe import AudioPipeline

def process_audio_folder(audio_folder, crop=False, random_crop=False, crop_length=5.0, num_slices=1, config=None):

    if not os.path.exists(audio_folder):
        raise ValueError(f"Audio folder '{audio_folder}' does not exist.")
    
    # Set audio pipeline
    if config:
        with open(config, 'r') as f:
            config_dict = json.load(f)
        pipeline = AudioPipeline.from_config(config_dict)
    else:
        pipeline = AudioPipeline()

    # File ops
    parent_folder = os.path.dirname(audio_folder)
    output_folder = os.path.join(parent_folder, "features")
    os.makedirs(output_folder, exist_ok=True)
    audio_files = [f for f in os.listdir(audio_folder) if f.endswith(('.wav', '.mp3', '.flac'))]

    for idx, audio_file in enumerate(audio_files):
        audio_path = os.path.join(audio_folder, audio_file)
        pipe_out = pipeline(audio_path)
        
        if "logmag" not in pipe_out:
            continue

        logmag = pipe_out["logmag"]
        
        # Cropping
        if crop:
            total_frames = logmag.shape[1]
            crop_samples = pipeline._time_to_frames(crop_length)

            if crop_samples > total_frames:

                # Zero padding if file shorter then crop length
                padding_needed = crop_samples - total_frames
                logmag = np.pad(logmag, ((0, 0), (0, padding_needed)), mode='constant')

                # Ignore multiple slicing
                num_slices = 1

            if random_crop and num_slices > 1:
                start_frames = [np.random.randint(0, total_frames - crop_samples) for _ in range(num_slices)]
            else:
                start_frames = [crop_samples * i for i in range(num_slices)]
        
        else:
            start_frames = [0]
            crop_samples = logmag.shape[1]

        for slice_idx, start_frame in enumerate(start_frames):

            # Take a spectrogram slice
            slice_data = logmag[:, start_frame:start_frame + crop_samples]

            # Name and save computed features
            hash_filename = hashlib.md5(audio_file.encode('utf-8')).hexdigest()
            slice_filename = f"{hash_filename}_{slice_idx:03d}.npy"
            slice_path = os.path.join(output_folder, slice_filename)
            np.save(slice_path, slice_data)
        
        print(f"{idx + 1} / {len(audio_files)} processed", end="\r")

    print(f"\nFinished processing {len(audio_files)} files. All features saved to '{output_folder}'.")

def main():

    parser = argparse.ArgumentParser(description="Preprocess audio files into logmag spectrogram slices.")
    
    parser.add_argument('audio_folder', type=str, help="Path to the folder containing audio files.")
    parser.add_argument('--crop', action='store_true', help="Whether to crop the audio files.")
    parser.add_argument('--random_crop', action='store_true', help="Whether to randomly crop or crop from the start.")
    parser.add_argument('--crop_length', type=float, default=5.0, help="Length of the crop in seconds.")
    parser.add_argument('--num_slices', type=int, default=1, help="How many slices to make from each per audio file.")
    parser.add_argument('--config', type=str, help="Path to the JSON config file for AudioPipeline.")

    args = parser.parse_args()

    process_audio_folder(
        audio_folder=args.audio_folder,
        crop=args.crop,
        random_crop=args.random_crop,
        crop_length=args.crop_length,
        num_slices=args.num_slices,
        config=args.config
    )

if __name__ == '__main__':
    main()