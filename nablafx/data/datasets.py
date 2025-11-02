import os
import sys
import glob
import torch
import torchaudio
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from natsort import natsorted


# -----------------------------------------------------------------------------
# Dataset classes for non-parametric and parametric models
# -----------------------------------------------------------------------------


class PluginDataset(torch.utils.data.Dataset):
    """
    Dataset of pre-rendered audio examples from VST plugin
    """

    def __init__(
        self,
        root_dir_dry: str,
        root_dir_wet: str,
        data_to_use: float = 1.0,
        sample_length: int = 48000,
        sample_rate: int = 48000,
        preload: bool = False,
    ):

        self.root_dir_dry = root_dir_dry
        self.root_dir_wet = root_dir_wet
        self.data_to_use = data_to_use
        self.sample_length = sample_length
        self.sample_rate = sample_rate
        self.preload = preload

        # get file paths
        self.input_files = glob.glob(os.path.join(self.root_dir_dry, "*.input.wav"))
        self.target_files = glob.glob(os.path.join(self.root_dir_wet, "*.target.wav"))

        # ensure that the sets are ordered correctly
        self.input_files = natsorted(self.input_files)
        self.target_files = natsorted(self.target_files)

        # check dry and wet files match
        for i, (input_file, target_file) in enumerate(zip(self.input_files, self.target_files)):
            ifile = os.path.basename(input_file).split(".")[-3]  # f"{filename}.input.wav"
            tfile = os.path.basename(target_file).split(".")[-3]  # f"{params_string}.{filename}.input.wav"
            if ifile != tfile:
                raise RuntimeError(f"Found non-matching files: {ifile} != {tfile}. Check dataset.")

        # get audio samples and params
        self.samples = []
        self.num_frames = 0  # total number of frames in the dataset

        # loop over files
        for idx, (ifile, tfile) in enumerate(zip(self.input_files, self.target_files)):
            print(ifile)
            print(tfile)

            md = torchaudio.info(tfile)
            num_frames = md.num_frames  # num samples
            self.num_frames += num_frames

            if self.preload:
                sys.stdout.write(f"* Pre-loading... {idx+1:3d}/{len(self.target_files):3d} ...\r")
                sys.stdout.flush()
                input, sr = self._load(ifile)
                target, sr = self._load(tfile)

                num_frames = int(np.min([input.shape[-1], target.shape[-1]]))
                if input.shape[-1] != target.shape[-1]:
                    print(
                        os.path.basename(ifile),
                        input.shape[-1],
                        os.path.basename(tfile),
                        target.shape[-1],
                    )
                    raise RuntimeError("Found potentially corrupt file!")
            else:
                input = None
                target = None
                sr = None

            # create one entry for each file
            self.file_samples = []
            if self.sample_length == -1:  # take whole file
                self.file_samples.append(
                    {
                        "idx": idx,
                        "input_file": ifile,
                        "target_file": tfile,
                        "input_audio": input if input is not None else None,
                        "target_audio": target if input is not None else None,
                        "offset": 0,
                        "frames": num_frames,
                        "sr": sr,
                    }
                )
            else:  # split into chunks
                for n in range((num_frames // self.sample_length)):
                    offset = int(n * self.sample_length)
                    end = offset + self.sample_length
                    self.file_samples.append(
                        {
                            "idx": idx,
                            "input_file": ifile,
                            "target_file": tfile,
                            "input_audio": (input[:, offset:end] if input is not None else None),
                            "target_audio": (target[:, offset:end] if input is not None else None),
                            "offset": offset,
                            "frames": num_frames,
                            "sr": sr,
                        }
                    )

            # add to overall file examples
            self.samples += self.file_samples

        # subset
        if data_to_use < 1.0:
            n = int(len(self.samples) * data_to_use)
            idxs = torch.randperm(len(self.samples))[:n]
            self.samples = [self.samples[i] for i in idxs]

        self.minutes = len(self.samples) * self.sample_length / self.sample_rate / 60.0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.preload:
            input = self.samples[idx]["input_audio"]
            target = self.samples[idx]["target_audio"]
        else:
            if self.sample_length == -1:  # whole file
                input, sr = self._load(self.samples[idx]["input_file"])
                target, sr = self._load(self.samples[idx]["target_file"])
            else:
                offset = self.samples[idx]["offset"]
                input, sr = self._load(
                    self.samples[idx]["input_file"],
                    frame_offset=offset,
                    num_frames=self.sample_length,
                )
                target, sr = self._load(
                    self.samples[idx]["target_file"],
                    frame_offset=offset,
                    num_frames=self.sample_length,
                )
        return input, target

    def _load(self, filepath: str, frame_offset: int = 0, num_frames: int = -1) -> Tuple[torch.Tensor, int]:
        x, sr = torchaudio.load(filepath, frame_offset, num_frames, normalize=True, channels_first=True)
        if sr != self.sample_rate:
            x = torchaudio.functional.resample(x, sr, self.sample_rate)
        return x, sr

    def print(self) -> None:
        print("\nPluginDataset:")
        print(f"num_samples: {len(self.samples)}")
        print(f"sample_length: {self.sample_length}")
        print(f"num_frames: {self.num_frames}")
        print(f"num_minutes: {self.minutes}")


# -----------------------------------------------------------------------------
# Dataset class for parametric models
# -----------------------------------------------------------------------------


class ParametricPluginDataset(torch.utils.data.Dataset):
    """
    Dataset of pre-rendered audio examples from VST plugin
    with associated parameters values
    """

    def __init__(
        self,
        root_dir_dry,
        root_dir_wet,
        params_idxs_to_use=None,
        data_to_use=1.0,
        sample_length=48000,
        sample_rate=48000,
        preload=False,
    ):

        self.root_dir_dry = root_dir_dry
        self.root_dir_wet = root_dir_wet
        self.params_idxs_to_use = params_idxs_to_use
        self.data_to_use = data_to_use
        self.sample_length = sample_length
        self.sample_rate = sample_rate
        self.preload = preload

        # get file paths
        self.input_files = glob.glob(os.path.join(self.root_dir_dry, "*.input.wav"))
        self.target_files = glob.glob(os.path.join(self.root_dir_wet, "*", "*.target.wav"))

        # ensure that the sets are ordered correctly
        self.input_files = natsorted(self.input_files)
        self.target_files = natsorted(self.target_files)

        # get audio samples and params
        self.samples = []
        self.num_frames = 0  # total number of frames in the dataset

        # loop over input files
        for iidx, ifile in enumerate(self.input_files):
            print(ifile)
            imd = torchaudio.info(ifile)

            # select corresponding target files
            ifilename = os.path.basename(ifile)[:-4]
            tfilename = ifilename.replace("input", "target")
            target_files = [t for t in self.target_files if tfilename in t]

            for tidx, tfile in enumerate(target_files):
                print(tfile)
                tmd = torchaudio.info(tfile)

                num_frames = int(np.min([imd.num_frames, tmd.num_frames]))
                self.num_frames += num_frames

                # extract params tensor from filename
                params = os.path.basename(tfile).split(".")[-4]  # get params string f"{params_string}.{filename}.input.wav"
                params = params.split("_")  # split params string f"{p1_letter}{p1_value}_{p2_letter}{p2_value}..."
                params = [float(p[1:]) / 100 for p in params]  # remove letter, convert to float, normalize to [0,1]
                params = torch.tensor(params)  # tensor
                params = params[self.params_idxs_to_use]  # select params to use

                if self.preload:
                    sys.stdout.write(f"* Pre-loading... {(iidx)*len(target_files)+tidx+1:3d}/{len(self.target_files):3d} ...\r")
                    sys.stdout.flush()
                    input, sr = self._load(ifile)
                    target, sr = self._load(tfile)
                else:
                    input = None
                    target = None
                    sr = None

                # one entry for each file or
                self.file_samples = []
                if self.sample_length == -1:  # take whole file
                    self.file_samples.append(
                        {
                            "iidx": iidx,
                            "tidx": tidx,
                            "input_file": ifile,
                            "target_file": tfile,
                            "input_audio": input if input is not None else None,
                            "target_audio": target if target is not None else None,
                            "params": params,
                            "offset": 0,
                            "frames": num_frames,
                            "sr": sr,
                        }
                    )
                # split into chunks
                else:
                    for n in range((num_frames // self.sample_length)):
                        offset = int(n * self.sample_length)
                        end = offset + self.sample_length
                        self.file_samples.append(
                            {
                                "iidx": iidx,
                                "tidx": tidx,
                                "input_file": ifile,
                                "target_file": tfile,
                                "input_audio": (input[:, offset:end] if input is not None else None),
                                "target_audio": (target[:, offset:end] if target is not None else None),
                                "params": params,
                                "offset": offset,
                                "frames": num_frames,
                                "sr": sr,
                            }
                        )
                # add to overall file examples
                self.samples += self.file_samples

        # subset
        if data_to_use < 1.0:
            n = int(len(self.samples) * data_to_use)
            idxs = torch.randperm(len(self.samples))[:n]
            self.samples = [self.samples[i] for i in idxs]

        self.minutes = len(self.samples) * self.sample_length / self.sample_rate / 60.0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.preload:
            input = self.samples[idx]["input_audio"]
            target = self.samples[idx]["target_audio"]
        else:
            if self.sample_length == -1:  # whole file
                input, sr = self._load(self.samples[idx]["input_file"])
                target, sr = self._load(self.samples[idx]["target_file"])
            else:
                offset = self.samples[idx]["offset"]
                input, sr = self._load(
                    self.samples[idx]["input_file"],
                    frame_offset=offset,
                    num_frames=self.sample_length,
                )
                target, sr = self._load(
                    self.samples[idx]["target_file"],
                    frame_offset=offset,
                    num_frames=self.sample_length,
                )

        # then get the tuple of parameters
        params = self.samples[idx]["params"]

        return input, target, params

    def _load(self, filepath: str, frame_offset: int = 0, num_frames: int = -1) -> Tuple[torch.Tensor, int]:
        x, sr = torchaudio.load(filepath, frame_offset, num_frames, normalize=True, channels_first=True)
        if sr != self.sample_rate:
            x = torchaudio.functional.resample(x, sr, self.sample_rate)
        return x, sr

    def print(self) -> None:
        print("\nParametricPluginDataset:")
        print(f"num_samples: {len(self.samples)}")
        print(f"sample_length: {self.sample_length}")
        print(f"num_frames: {self.num_frames}")
        print(f"num_minutes: {self.minutes}")
        print(f"params_idxs_to_use: {self.params_idxs_to_use}")


if __name__ == "__main__":
    dataset = PluginDataset(
        root_dir_dry="/Volumes/BUTCH/DATASETS/NNLIN-AFX-DATASET-NEW-STRUCTURE/ANALOG/DRY-with-markers/trainval",
        root_dir_wet="/Volumes/BUTCH/DATASETS/NNLIN-AFX-DATASET-NEW-STRUCTURE/ANALOG/Ampeg-OptoComp/trainval/C030_R050_L060",
        data_to_use=1.0,
        sample_length=480000,
        sample_rate=48000,
        preload=False,
    )
    dataset.print()
    input, target = dataset[0]
    print(input.shape, target.shape)

    print()
    dataset = ParametricPluginDataset(
        root_dir_dry="/Volumes/BUTCH/DATASETS/NNLIN-AFX-DATASET-NEW-STRUCTURE/ANALOG-EXTERNAL/stepan-miklanek-greyboxamp/Marshall-JVM410H-ChOD1/DRY/trainval",
        root_dir_wet="/Volumes/BUTCH/DATASETS/NNLIN-AFX-DATASET-NEW-STRUCTURE/ANALOG-EXTERNAL/stepan-miklanek-greyboxamp/Marshall-JVM410H-ChOD1/PreampOut/trainval",
        # params_file="/Volumes/GATSBY/DATASETS/NNLIN-AFX-DATASET/AFX/DIGITAL-PARAMETRIC/MultidrivePedalPro-808-Scream/settings.csv",
        params_idxs_to_use=[0, 1],
        data_to_use=1.0,
        sample_length=480000,
        sample_rate=48000,
        preload=False,
    )
    dataset.print()
    input, target, params = dataset[0]
    print(input.shape, target.shape, params)

    input, target, params = dataset[50]
    print(input.shape, target.shape, params)

    input, target, params = dataset[100]
    print(input.shape, target.shape, params)

    input, target, params = dataset[150]
    print(input.shape, target.shape, params)

    input, target, params = dataset[200]
    print(input.shape, target.shape, params)
