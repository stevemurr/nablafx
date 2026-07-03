import os
import sys
import glob
import torch
import torchaudio
import soundfile as sf
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from natsort import natsorted


def _audio_num_frames(path: str) -> int:
    # torchaudio.info was removed in torchaudio >= 2.8; soundfile is
    # already a transitive dep and gives us frame counts cheaply.
    return sf.info(path).frames


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
        gain_aug_db: Optional[List[float]] = None,
        train: bool = True,
    ):

        self.root_dir_dry = root_dir_dry
        self.root_dir_wet = root_dir_wet
        self.data_to_use = data_to_use
        self.sample_length = sample_length
        self.sample_rate = sample_rate
        self.preload = preload
        if gain_aug_db is None:
            self.gain_aug_db = None
        else:
            lo, hi = float(gain_aug_db[0]), float(gain_aug_db[1])
            if lo > hi:
                raise ValueError(f"gain_aug_db must be [low, high] with low <= high, got [{lo}, {hi}]")
            self.gain_aug_db = (lo, hi)
        self.train = train

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

            num_frames = _audio_num_frames(tfile)
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
        if self.train and self.gain_aug_db is not None:
            lo, hi = self.gain_aug_db
            g_db = lo + torch.rand(1).item() * (hi - lo)
            g = 10.0 ** (g_db / 20.0)
            input = input * g
            target = target * g
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
# Dataset class for the SSL console EQ (params from an npz sidecar, not filenames)
# -----------------------------------------------------------------------------


class SSLParametricPluginDataset(torch.utils.data.Dataset):
    """Paired (dry, wet) MONO examples for the knob-conditioned SSL console EQ.

    Layout produced by neural-mastering/scripts/prepare_ssl_eq_data.py:
        <dry>/{name}.input.wav   <wet>/{name}.target.wav
        <sidecar>/{split}.npz  -> names, cond (N, C), tf_mag_db (N, 256)

    Unlike ParametricPluginDataset (which parses lossy params from filenames),
    conditioning comes from the npz sidecar keyed by clip name — full float
    precision for the ~20-dim physical vector. Returns (input, target, params).
    The measured transfer function per name is kept in ``self.tf_by_name`` for the
    Phase-3 TF-matching loss (not returned in the batch to preserve the 3-tuple
    grey-box contract).
    """

    def __init__(
        self,
        root_dir_dry: str,
        root_dir_wet: str,
        params_sidecar: str,
        data_to_use: float = 1.0,
        sample_length: int = 144000,
        sample_rate: int = 48000,
        preload: bool = False,
        gain_aug_db: Optional[List[float]] = None,
        train: bool = True,
    ):
        self.root_dir_dry = root_dir_dry
        self.root_dir_wet = root_dir_wet
        self.sample_length = sample_length
        self.sample_rate = sample_rate
        self.preload = preload
        self.train = train
        self.gain_aug_db = None if gain_aug_db is None else (float(gain_aug_db[0]), float(gain_aug_db[1]))

        side = np.load(params_sidecar, allow_pickle=True)
        names = [str(n) for n in side["names"]]
        cond = side["cond"].astype(np.float32)
        tf = side["tf_mag_db"].astype(np.float32)
        self.cond_by_name = {n: cond[i] for i, n in enumerate(names)}
        self.tf_by_name = {n: tf[i] for i, n in enumerate(names)}
        self.num_controls = cond.shape[1]

        self.input_files = natsorted(glob.glob(os.path.join(root_dir_dry, "*.input.wav")))
        self.target_files = natsorted(glob.glob(os.path.join(root_dir_wet, "*.target.wav")))

        self.samples = []
        self.num_frames = 0
        for idx, (ifile, tfile) in enumerate(zip(self.input_files, self.target_files)):
            name = os.path.basename(ifile).split(".")[-3]
            if name != os.path.basename(tfile).split(".")[-3]:
                raise RuntimeError(f"dry/wet mismatch: {ifile} vs {tfile}")
            if name not in self.cond_by_name:
                raise RuntimeError(f"no conditioning for {name} in {params_sidecar}")
            params = torch.from_numpy(self.cond_by_name[name])
            num_frames = _audio_num_frames(tfile)
            self.num_frames += num_frames
            inp = tgt = None
            if preload:
                inp, _ = self._load(ifile)
                tgt, _ = self._load(tfile)
                num_frames = int(min(inp.shape[-1], tgt.shape[-1]))
            chunks = [(0, -1)] if sample_length == -1 else \
                [(n * sample_length, (n + 1) * sample_length) for n in range(num_frames // sample_length)]
            for off, end in chunks:
                self.samples.append(dict(
                    name=name, input_file=ifile, target_file=tfile, offset=off, params=params,
                    input_audio=(inp if off == 0 and end == -1 else (inp[:, off:end] if inp is not None else None)),
                    target_audio=(tgt if off == 0 and end == -1 else (tgt[:, off:end] if tgt is not None else None)),
                ))

        if data_to_use < 1.0:
            n = int(len(self.samples) * data_to_use)
            idxs = torch.randperm(len(self.samples))[:n]
            self.samples = [self.samples[i] for i in idxs]
        self.minutes = len(self.samples) * self.sample_length / self.sample_rate / 60.0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        if self.preload:
            input, target = s["input_audio"], s["target_audio"]
        elif self.sample_length == -1:
            input, _ = self._load(s["input_file"])
            target, _ = self._load(s["target_file"])
        else:
            input, _ = self._load(s["input_file"], s["offset"], self.sample_length)
            target, _ = self._load(s["target_file"], s["offset"], self.sample_length)
        if self.train and self.gain_aug_db is not None:
            lo, hi = self.gain_aug_db
            g = 10.0 ** ((lo + torch.rand(1).item() * (hi - lo)) / 20.0)
            input, target = input * g, target * g
        return input, target, s["params"]

    def _load(self, filepath, frame_offset=0, num_frames=-1):
        x, sr = torchaudio.load(filepath, frame_offset, num_frames, normalize=True, channels_first=True)
        if sr != self.sample_rate:
            x = torchaudio.functional.resample(x, sr, self.sample_rate)
        return x, sr

    def print(self):
        print(f"\nSSLParametricPluginDataset: {len(self.samples)} samples, "
              f"{self.num_controls} controls, {self.minutes:.1f} min")


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
            imd_frames = _audio_num_frames(ifile)

            # select corresponding target files
            ifilename = os.path.basename(ifile)[:-4]
            tfilename = ifilename.replace("input", "target")
            target_files = [t for t in self.target_files if tfilename in t]

            for tidx, tfile in enumerate(target_files):
                print(tfile)
                tmd_frames = _audio_num_frames(tfile)

                num_frames = int(np.min([imd_frames, tmd_frames]))
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
