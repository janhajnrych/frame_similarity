import argparse
import cv2
from tqdm import tqdm
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt


class VideoSearcher:

    def __init__(self, threshold, horizon, n_zones):
        self.hasher = cv2.img_hash.BlockMeanHash_create()
        self.Nx = n_zones
        self.Ny = n_zones
        self.hash_map_list = []
        self.threshold = threshold
        self.horizon = horizon

    @staticmethod
    def get_props(cap):
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return width, height, fps, length

    @staticmethod
    def create_writer(out_path, width, height, fps):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (2 * width, height))
        return out

    def _calc_hash(self, frame):
        dx = frame.shape[1] // self.Nx
        dy = frame.shape[0] // self.Ny
        hash_map = np.zeros((self.Ny, self.Nx, 32))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for ix in range(self.Nx):
            for iy in range(self.Ny):
                subimg = gray[iy * dy: (iy + 1) * dy, ix * dx: (ix + 1) * dx]
                h = self.hasher.compute(subimg)
                hash_map[iy, ix, :] = h
        return hash_map

    def generate_hashes(self, ref_video_file_path):
        cap = cv2.VideoCapture(ref_video_file_path)
        width, height, fps, length = self.get_props(cap)
        for i in tqdm(range(length), desc="generating hashes"):
            success, frame = cap.read()
            if not success:
                continue
            ref_hash_map = self._calc_hash(frame)
            self.hash_map_list.append((ref_hash_map, i))

    def find_similar(self, frame, n_frames, start_index=0):
        end_index = min(n_frames-1, start_index + self.horizon)
        ref_hash_list = self.hash_map_list[start_index:end_index]
        frame_hash_map = self._calc_hash(frame)
        distances = []
        frame_indexes = []
        for ref_hash_map, i in ref_hash_list:
            d = distance.hamming(ref_hash_map.flatten(), frame_hash_map.flatten())
            distances.append(d)
            frame_indexes.append(i)
        distances = np.array(distances)
        min_distance = np.min(distances)
        return frame_indexes[np.argmin(distances)], min_distance

    def process_movie(self, in_video_file_path, ref_video_file_path, out_path):
        cap = cv2.VideoCapture(in_video_file_path)
        cap_ref = cv2.VideoCapture(ref_video_file_path)
        width, height, fps, length = self.get_props(cap)
        ref_width, ref_height, ref_fps, ref_length = self.get_props(cap)
        out = self.create_writer(out_path, width, height, fps)
        last_found_id = 0
        distance_matrix = np.ones((length, ref_length))
        for i in tqdm(range(length), desc="analyzing"):
            success, frame = cap.read()
            if not success:
                continue
            ref_frame_id, distance = self.find_similar(frame, length, start_index=last_found_id)
            distance_matrix[i, ref_frame_id] = distance
            if distance > self.threshold:
                ref_frame = np.zeros((height, width, 3), np.uint8)
            else:
                last_found_id = ref_frame_id
                cap_ref.set(cv2.CAP_PROP_POS_FRAMES, ref_frame_id)
                ref_success, ref_frame = cap_ref.read()
            out_img = np.concatenate((frame, ref_frame), axis=1)
            out.write(out_img)
        out.release()
        cap.release()
        cap_ref.release()
        return distance_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("program", help="program video")
    parser.add_argument("reference", help="reference video")
    parser.add_argument("output", help="output path")
    parser.add_argument("-t", "--thresh", help="distance threshold", type= float, default=0.5)
    parser.add_argument("-x", "--horizon", help="frame horizon", type=int, default=16)
    parser.add_argument("-z", "--zones", help="number of zones", type=int, default=16)
    parser.add_argument("--plot", help="correspondence plot path", default=None)
    args = parser.parse_args()
    if args.thresh < 0.0:
        raise ValueError(f"thresh must be positive, got {args.thresh}")
    if args.thresh > 1.0:
        raise ValueError(f"thresh must be below 1.0, got {args.thresh}")
    searcher = VideoSearcher(args.thresh, args.horizon, args.zones)
    searcher.generate_hashes(args.reference)
    distance_matrix = searcher.process_movie(args.program, args.reference, args.output)
    plt.imshow(distance_matrix, cmap='hot', interpolation='nearest')
    if args.plot is not None:
        plt.colorbar()
        plt.title("frame correspondence")
        plt.savefig(args.plot)