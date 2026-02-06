import h5py
record_file_name = f"./raw/episode_000_20260121_141810.h5"

with h5py.File(record_file_name, "r") as data:
    print(data.keys())
    print("color length:", data["color"].shape)